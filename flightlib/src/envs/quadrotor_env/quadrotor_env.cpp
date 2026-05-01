#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(), current_gate_idx_(0), num_gates_(0),
    s_prev_(0.0), passed_gate_this_step_(false) {
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);
  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  world_box_ << -50, 50, -50, 50, 0, 50;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set world box");
  };

  obs_dim_ = quadenv::kNObs;  // 26
  act_dim_ = quadenv::kNAct;  // 4

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  loadParam(cfg_);
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random && num_gates_ > 0) {
    // ══════════════════════════════════════════════════════════════
    // IROS 2021 Distributed Initialization (Paper L247-261)
    //
    // Phase 1: "We randomly initialize the quadrotors in a hover state
    //   around the centers of all path segments" (L257)
    //
    // Phase 2: "Once the policy has learned to pass gates reliably,
    //   we sample initial states from previous trajectories" (L259)
    //   "Thus, we retain the benefit of initializing across the whole
    //   track while avoiding the negative impact of starting from a
    //   hover position in areas of the track that are associated with
    //   high speeds." (L261)
    // ══════════════════════════════════════════════════════════════

    // Check if we should transition to Phase 2.
    // mean_gates_per_ep is total_gates_passed / total_episodes — i.e. the
    // average number of gates cleared per episode (NOT a probability).
    // The paper says "once the policy passes gates reliably" (L259); we
    // approximate "reliably" as a threshold on this mean.
    if (!use_traj_init_ && total_episodes_ > traj_init_min_episodes_) {
      Scalar mean_gates_per_ep = Scalar(total_gates_passed_) /
                                 std::max(Scalar(total_episodes_), Scalar(1.0));
      if (mean_gates_per_ep > traj_init_threshold_ && traj_buffer_.size() > 100) {
        use_traj_init_ = true;
        logger_.info("Switching to Phase 2 init (traj replay), "
                     "mean_gates_per_ep=" + std::to_string(mean_gates_per_ep));
      }
    }

    bool used_traj = false;
    // Phase 2: sample from trajectory buffer with traj_init_fraction_ probability
    if (use_traj_init_ && !traj_buffer_.empty() &&
        uniform_dist_(random_gen_) < traj_init_fraction_) {
      // Pick random snapshot from buffer
      std::uniform_int_distribution<int> idx_dist(0, traj_buffer_.size() - 1);
      const auto& snap = traj_buffer_[idx_dist(random_gen_)];

      // Set state from snapshot with small perturbation
      quad_state_.x(QS::POSX) = snap.pos(0) + Scalar(uniform_dist_(random_gen_) * 0.3);
      quad_state_.x(QS::POSY) = snap.pos(1) + Scalar(uniform_dist_(random_gen_) * 0.3);
      Scalar pz_snap = snap.pos(2) + Scalar(uniform_dist_(random_gen_) * 0.3);
      quad_state_.x(QS::POSZ) = std::max(pz_snap, Scalar(0.5));
      quad_state_.x(QS::VELX) = snap.vel(0) + Scalar(uniform_dist_(random_gen_) * 0.5);
      quad_state_.x(QS::VELY) = snap.vel(1) + Scalar(uniform_dist_(random_gen_) * 0.5);
      quad_state_.x(QS::VELZ) = snap.vel(2) + Scalar(uniform_dist_(random_gen_) * 0.3);
      // Use snapshot orientation with small perturbation
      Quaternion q = snap.quat;
      q.w() += uniform_dist_(random_gen_) * 0.05;
      q.x() += uniform_dist_(random_gen_) * 0.05;
      q.y() += uniform_dist_(random_gen_) * 0.05;
      q.z() += uniform_dist_(random_gen_) * 0.05;
      q.normalize();
      quad_state_.x(QS::ATTW) = q.w();
      quad_state_.x(QS::ATTX) = q.x();
      quad_state_.x(QS::ATTY) = q.y();
      quad_state_.x(QS::ATTZ) = q.z();
      // Restore body angular velocity from snapshot (was being dropped,
      // which left ω=0 and partially recreated the hover-init pathology
      // the paper warns about for high-speed regions).
      quad_state_.w(0) = snap.omega(0) + Scalar(uniform_dist_(random_gen_) * 0.2);
      quad_state_.w(1) = snap.omega(1) + Scalar(uniform_dist_(random_gen_) * 0.2);
      quad_state_.w(2) = snap.omega(2) + Scalar(uniform_dist_(random_gen_) * 0.2);
      current_gate_idx_ = snap.gate_idx;
      used_traj = true;
    }

    // Phase 1 (or Phase 2 fallback): Hover init across all segments
    if (!used_traj) {
      // Pick a random segment uniformly (Paper L257: "centers of all path segments")
      std::uniform_int_distribution<int> seg_dist(0, num_gates_ - 1);
      int seg_idx = seg_dist(random_gen_);
      current_gate_idx_ = seg_idx;

      // Midpoint of segment (from previous gate to this gate)
      // Segment 0 uses start_pos_ as origin (track is not circular)
      Vector<3> g_prev = (seg_idx == 0) ? start_pos_ :
                          gates_[(seg_idx - 1 + num_gates_) % num_gates_];
      Vector<3> g_curr = gates_[seg_idx];
      // Paper L257: spawn "around the centers of all path segments".
      // uniform_dist_ ∈ [-1, 1], so this gives t ∈ [0.2, 0.8] centered on 0.5.
      Scalar t = Scalar(0.5) + Scalar(uniform_dist_(random_gen_)) * Scalar(0.3);
      Vector<3> spawn = g_prev + t * (g_curr - g_prev);

      quad_state_.x(QS::POSX) = spawn(0) + Scalar(uniform_dist_(random_gen_)) * Scalar(1.5);
      quad_state_.x(QS::POSY) = spawn(1) + Scalar(uniform_dist_(random_gen_)) * Scalar(1.5);
      Scalar pz_seg = spawn(2) + Scalar(uniform_dist_(random_gen_));
      quad_state_.x(QS::POSZ) = std::max(pz_seg, Scalar(0.5));
      // Hover: near-zero velocity (Paper L257: "hover state")
      quad_state_.x(QS::VELX) = Scalar(uniform_dist_(random_gen_)) * Scalar(0.5);
      quad_state_.x(QS::VELY) = Scalar(uniform_dist_(random_gen_)) * Scalar(0.5);
      quad_state_.x(QS::VELZ) = Scalar(uniform_dist_(random_gen_)) * Scalar(0.3);
      // Upright with small perturbation
      quad_state_.x(QS::ATTW) = 1.0;
      quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_) * 0.1;
      quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_) * 0.1;
      quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_) * 0.1;
      quad_state_.qx /= quad_state_.qx.norm();
    }
  } else if (random) {
    // No gates: random position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = std::abs(uniform_dist_(random_gen_)) + 5;
    quad_state_.x(QS::ATTW) = 1.0;
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_) * 0.1;
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_) * 0.1;
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_) * 0.1;
    quad_state_.qx /= quad_state_.qx.norm();
  } else {
    // Deterministic reset (evaluation): start at start_pos_ in hover,
    // facing toward the first gate
    quad_state_.x(QS::POSX) = start_pos_(0);
    quad_state_.x(QS::POSY) = start_pos_(1);
    quad_state_.x(QS::POSZ) = start_pos_(2);
    quad_state_.x(QS::ATTW) = 1.0;
    quad_state_.x(QS::ATTX) = 0.0;
    quad_state_.x(QS::ATTY) = 0.0;
    quad_state_.x(QS::ATTZ) = 0.0;
    current_gate_idx_ = 0;
  }

  quadrotor_ptr_->reset(quad_state_);
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // Reset episode tracking
  passed_gate_this_step_ = false;
  step_counter_ = 0;
  gates_passed_this_ep_ = 0;
  if (num_gates_ > 0) {
    s_prev_ = spline_progress(quad_state_.p, current_gate_idx_);
  } else {
    s_prev_ = 0.0;
  }
  prev_vel_ = quad_state_.v;
  prev_pos_for_gate_detect_ = quad_state_.p;

  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // ── 1. Linear velocity (3) ──
  quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel) = quad_state_.v;

  // ── 2. Linear acceleration via finite difference (3) ──
  Vector<3> lin_acc = (quad_state_.v - prev_vel_) / sim_dt_;
  quad_obs_.segment<quadenv::kNLinAcc>(quadenv::kLinAcc) = lin_acc;

  // ── 3. Rotation matrix, flattened row-major (9) ──
  Matrix<3, 3> R_wb = quad_state_.q().toRotationMatrix();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      quad_obs_(quadenv::kRotMat + r * 3 + c) = R_wb(r, c);

  // ── 4. Angular velocity (3) ──
  quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) = quad_state_.w;

  // ── 5. Gate observations: 2 future gates × 4 dims each = 8 ──
  if (num_gates_ > 0) {
    // Gate 1 (next gate): spherical in body frame
    int g1_idx = current_gate_idx_;
    Vector<3> g1_world = gates_[g1_idx];
    Vector<3> rel1_world = g1_world - quad_state_.p;
    Vector<3> rel1_body = R_wb.transpose() * rel1_world;

    Scalar r1 = rel1_body.norm();
    Scalar theta1 = Scalar(0), phi1 = Scalar(0);
    if (r1 > Scalar(1e-6)) {
      theta1 = std::atan2(rel1_body(1), rel1_body(0));
      Scalar sp = std::max(std::min(rel1_body(2) / r1, Scalar(1)), Scalar(-1));
      phi1 = std::asin(sp);
    }

    Scalar alpha1 = Scalar(0);
    Scalar r1w = rel1_world.norm();
    if (r1w > Scalar(1e-6)) {
      Scalar ca = std::max(std::min(gate_normals_[g1_idx].dot(rel1_world / r1w),
                                     Scalar(1)), Scalar(-1));
      alpha1 = std::acos(ca);
    }

    int b1 = quadenv::kGateObs;
    quad_obs_(b1) = r1; quad_obs_(b1+1) = theta1;
    quad_obs_(b1+2) = phi1; quad_obs_(b1+3) = alpha1;

    // Gate 2 (2nd future): spherical in gate 1's frame (paper L189)
    int g2_idx = (current_gate_idx_ + 1) % num_gates_;
    Vector<3> rel2_g1w = gates_[g2_idx] - g1_world;
    Vector<3> rel2_g1f = gate_frames_[g1_idx].transpose() * rel2_g1w;

    Scalar r2 = rel2_g1f.norm();
    Scalar theta2 = Scalar(0), phi2 = Scalar(0);
    if (r2 > Scalar(1e-6)) {
      theta2 = std::atan2(rel2_g1f(1), rel2_g1f(0));
      Scalar sp2 = std::max(std::min(rel2_g1f(2) / r2, Scalar(1)), Scalar(-1));
      phi2 = std::asin(sp2);
    }

    Scalar alpha2 = Scalar(0);
    Vector<3> rel2_dw = gates_[g2_idx] - quad_state_.p;
    Scalar r2d = rel2_dw.norm();
    if (r2d > Scalar(1e-6)) {
      Scalar ca2 = std::max(std::min(gate_normals_[g2_idx].dot(rel2_dw / r2d),
                                      Scalar(1)), Scalar(-1));
      alpha2 = std::acos(ca2);
    }

    int b2 = quadenv::kGateObs + quadenv::kNPerGate;
    quad_obs_(b2) = r2; quad_obs_(b2+1) = theta2;
    quad_obs_(b2+2) = phi2; quad_obs_(b2+3) = alpha2;
  } else {
    quad_obs_.segment<quadenv::kNGateObs>(quadenv::kGateObs).setZero();
  }

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;
  prev_vel_ = quad_state_.v;
  quadrotor_ptr_->run(cmd_, sim_dt_);
  getObs(obs);

  // ── Trajectory snapshot recording (for Phase 2 init) ──
  step_counter_++;
  if (num_gates_ > 0 && step_counter_ % traj_snapshot_interval_ == 0) {
    // Only record states with reasonable velocity (flying, not crashed)
    if (quad_state_.v.norm() > Scalar(0.5) && quad_state_.x(QS::POSZ) > Scalar(0.5)) {
      TrajectorySnapshot snap;
      snap.pos = quad_state_.p;
      snap.vel = quad_state_.v;
      snap.quat = quad_state_.q();
      snap.omega = quad_state_.w;
      snap.gate_idx = current_gate_idx_;
      traj_buffer_.push_back(snap);
      if (static_cast<int>(traj_buffer_.size()) > traj_buffer_max_) {
        traj_buffer_.pop_front();
      }
    }
  }

  // ── IROS 2021 REWARD (Paper Eq. 2-5) ──
  passed_gate_this_step_ = false;
  Scalar total_reward = 0.0;

  // Reset per-step reward component tracking
  last_progress_reward_ = 0.0;
  last_safety_reward_ = 0.0;
  last_body_rate_penalty_ = 0.0;
  last_deviation_penalty_ = 0.0;

  if (num_gates_ > 0) {
    // 1. Progress Reward (Paper Eq. 2): r_p = s(p_t) - s(p_{t-1}), unclamped.
    //    Uses spline-based progress for correct curved-path tracking.
    Scalar s_curr = spline_progress(quad_state_.p, current_gate_idx_);
    Scalar progress = s_curr - s_prev_;
    last_progress_reward_ = progress;
    total_reward += progress;

    // 1b. Per-step deviation penalty: quadratic cost for straying beyond gate radius.
    // Unlike the linear penalty inside spline_progress (which only fires on
    // deviation *changes*), this fires every step the drone is far from the
    // track, preventing it from cruising parallel at a large constant offset.
    if (deviation_penalty_coeff_ > 0.0) {
      Scalar dev = spline_nearest_dist(quad_state_.p, current_gate_idx_);
      Scalar threshold = gate_width_ / Scalar(2.0);
      if (dev > threshold) {
        Scalar excess = dev - threshold;
        last_deviation_penalty_ = -deviation_penalty_coeff_ * excess * excess;
        total_reward += last_deviation_penalty_;
      }
    }

    // 2. Safety Reward (Paper Eq. 2-3)
    Vector<3> g_curr = gates_[current_gate_idx_];
    Vector<3> n_curr = gate_normals_[current_gate_idx_];
    Scalar d_n = (quad_state_.p - g_curr).dot(n_curr); // distance to gate plane
    Scalar d_p = (quad_state_.p - g_curr - d_n * n_curr).norm(); // distance to gate axis
    
    if (safety_a_ > 0.0 && d_p < safety_d_max_) {
      Scalar f = std::max(Scalar(1.0) - (d_p / safety_d_max_), Scalar(0.0));
      Scalar v = std::max((Scalar(1.0) - f) * (gate_width_ / Scalar(6.0)), Scalar(0.05));
      Scalar r_s = -f * f * (Scalar(1.0) - std::exp(-Scalar(0.5) * d_n * d_n / v));
      last_safety_reward_ = safety_a_ * r_s;
      total_reward += last_safety_reward_;
    }

    // 3. Gate Pass Detection
    // Use gate-plane crossing with proximity guard: d_n transitions from
    // negative to positive when the drone is near the gate.
    // No gate collision termination — the agent is only rewarded for
    // progress along the spline. Gate pass detection advances the
    // target gate index so the spline progress tracks correctly.
    Scalar d_n_prev = (prev_pos_for_gate_detect_ - g_curr).dot(n_curr);
    Scalar dist_to_gate = (quad_state_.p - g_curr).norm();
    Scalar gate_proximity = gate_width_ * Scalar(2.0);

    if (d_n_prev < Scalar(0) && d_n >= Scalar(0) &&
        dist_to_gate < gate_proximity) {
      // Interpolate to the exact plane-crossing position
      Scalar denom = d_n_prev - d_n;
      Scalar t_cross = (std::abs(denom) > Scalar(1e-9))
                           ? (d_n_prev / denom)
                           : Scalar(0.5);
      Vector<3> cross_pos = prev_pos_for_gate_detect_ +
                            t_cross * (quad_state_.p - prev_pos_for_gate_detect_);
      Vector<3> offset = cross_pos - g_curr;

      // Rectangular gate check: project offset onto gate right/up axes
      Vector<3> gate_right = gate_frames_[current_gate_idx_].col(1);
      Vector<3> gate_up    = gate_frames_[current_gate_idx_].col(2);
      Scalar off_right = std::abs(offset.dot(gate_right));
      Scalar off_up    = std::abs(offset.dot(gate_up));
      Scalar half_w = gate_width_ / Scalar(2.0);

      if (off_right < half_w && off_up < half_w) {
        // SUCCESS: Through the gate opening — advance to next gate.
        passed_gate_this_step_ = true;
        gates_passed_this_ep_++;
        total_gates_passed_++;
        current_gate_idx_ = (current_gate_idx_ + 1) % num_gates_;
        s_prev_ = spline_progress(quad_state_.p, current_gate_idx_);
      }
      // If outside the opening, do nothing — no collision penalty.
    } else {
      s_prev_ = s_curr;
    }
    prev_pos_for_gate_detect_ = quad_state_.p;
  }

  // 4. Body Rate Penalty with Annealing (Paper Eq. 4)
  total_steps_++;
  Scalar b = body_rate_coeff_;
  if (body_rate_anneal_steps_ > 0) {
    Scalar factor = std::max(Scalar(1.0) - Scalar(total_steps_) / body_rate_anneal_steps_, Scalar(0.0));
    b *= factor;
  }
  if (b > 0.0) {
    last_body_rate_penalty_ = -b * quad_state_.w.squaredNorm();
    total_reward += last_body_rate_penalty_;
  }

  // Reward = progress + safety + body-rate penalty.
  // No gate collision penalty or termination — only progress matters.

  last_total_reward_ = total_reward;
  return total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  // Terminal conditions: ground crash, out of bounds, or timeout.
  // No gate collision termination — the agent only needs to make
  // progress along the track spline.
  //
  // term_reason_ encoding (exposed via extra_info for diagnostics):
  //   0 = running (not terminal)
  //   2 = ground crash (z <= 0.02)
  //   3 = out of bounds (||p|| > 100)
  //   4 = timeout (t >= max_t)
  bool terminal = false;
  term_reason_ = 0;
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = 0.0; terminal = true; term_reason_ = 2;
  } else if (quad_state_.p.norm() > 100.0) {
    reward = 0.0; terminal = true; term_reason_ = 3;
  } else if (cmd_.t >= max_t_) {
    reward = 0.0; terminal = true; term_reason_ = 4;
  } else {
    reward = 0.0;
  }
  // FIX: Only count episodes when they actually terminate.
  // Previously this ran every step (isTerminalState is called per-step
  // by VecEnv), making total_episodes_ count steps instead of episodes,
  // which broke the Phase 2 initialization transition logic.
  if (terminal) {
    total_episodes_++;
  }
  return terminal;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    logger_.error("Missing quadrotor_env config section");
    return false;
  }

  // ── Load start position ──
  if (cfg["start_pos"]) {
    auto sp = cfg["start_pos"];
    start_pos_ << sp[0].as<Scalar>(), sp[1].as<Scalar>(), sp[2].as<Scalar>();
    logger_.info("Start position: [" +
                 std::to_string(start_pos_(0)) + ", " +
                 std::to_string(start_pos_(1)) + ", " +
                 std::to_string(start_pos_(2)) + "]");
  }

  // ── Load gate positions and orientations ──
  if (cfg["gates"]) {
    gates_.clear(); gate_normals_.clear(); gate_frames_.clear();
    for (const auto &gate_node : cfg["gates"]) {
      auto pos = gate_node["position"];
      Vector<3> gate_pos;
      gate_pos << pos[0].as<Scalar>(), pos[1].as<Scalar>(), pos[2].as<Scalar>();
      gates_.push_back(gate_pos);

      // ── Parse gate rotation → full 3x3 rotation matrix ──
      // Supports two formats:
      //   rotation_quat: [w, x, y, z]  — quaternion (paper's original format)
      //   rotation: [yaw, pitch, roll]  — ZYX Euler in degrees (our format)
      // The gate normal (fly-through direction) = R * [1, 0, 0]
      Matrix<3, 3> R_gate = Matrix<3, 3>::Identity();

      if (gate_node["rotation_quat"]) {
        // Quaternion format [w, x, y, z] — used by original IROS 2021 track files
        auto rq = gate_node["rotation_quat"];
        Scalar qw = rq[0].as<Scalar>();
        Scalar qx = rq[1].as<Scalar>();
        Scalar qy = rq[2].as<Scalar>();
        Scalar qz = rq[3].as<Scalar>();
        // Normalize
        Scalar qn = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qn > Scalar(1e-8)) { qw /= qn; qx /= qn; qy /= qn; qz /= qn; }
        // Quaternion to rotation matrix
        R_gate(0,0) = Scalar(1) - Scalar(2)*(qy*qy + qz*qz);
        R_gate(0,1) = Scalar(2)*(qx*qy - qw*qz);
        R_gate(0,2) = Scalar(2)*(qx*qz + qw*qy);
        R_gate(1,0) = Scalar(2)*(qx*qy + qw*qz);
        R_gate(1,1) = Scalar(1) - Scalar(2)*(qx*qx + qz*qz);
        R_gate(1,2) = Scalar(2)*(qy*qz - qw*qx);
        R_gate(2,0) = Scalar(2)*(qx*qz - qw*qy);
        R_gate(2,1) = Scalar(2)*(qy*qz + qw*qx);
        R_gate(2,2) = Scalar(1) - Scalar(2)*(qx*qx + qy*qy);
      } else if (gate_node["rotation"]) {
        // Euler format [yaw, pitch, roll] in degrees (ZYX convention)
        auto rot = gate_node["rotation"];
        Scalar yaw   = rot[0].as<Scalar>() * Scalar(M_PI) / Scalar(180.0);
        Scalar pitch = rot[1].as<Scalar>() * Scalar(M_PI) / Scalar(180.0);
        Scalar roll  = rot[2].as<Scalar>() * Scalar(M_PI) / Scalar(180.0);
        // R = Rz(yaw) * Ry(pitch) * Rx(roll)
        Scalar cy = std::cos(yaw), sy = std::sin(yaw);
        Scalar cp = std::cos(pitch), sp = std::sin(pitch);
        Scalar cr = std::cos(roll), sr = std::sin(roll);
        R_gate(0,0) = cy*cp;  R_gate(0,1) = cy*sp*sr - sy*cr;  R_gate(0,2) = cy*sp*cr + sy*sr;
        R_gate(1,0) = sy*cp;  R_gate(1,1) = sy*sp*sr + cy*cr;  R_gate(1,2) = sy*sp*cr - cy*sr;
        R_gate(2,0) = -sp;    R_gate(2,1) = cp*sr;              R_gate(2,2) = cp*cr;
      }

      // Gate normal = first column of R_gate (the rotated X-axis = fly-through direction)
      Vector<3> normal = R_gate.col(0);
      normal.normalize();
      gate_normals_.push_back(normal);

      // Store the full gate frame for gate-2 obs (paper L189: recursive frame)
      gate_frames_.push_back(R_gate);
    }
    num_gates_ = static_cast<int>(gates_.size());
    logger_.info("Loaded " + std::to_string(num_gates_) + " gates with normals");
  } else {
    logger_.warn("No gates defined in config — running in hover mode");
    num_gates_ = 0;
  }

  // ── Load reward coefficients (Paper Eq. 4-5) ──
  if (cfg["rl"]) {
    if (cfg["rl"]["body_rate_coeff"]) body_rate_coeff_ = cfg["rl"]["body_rate_coeff"].as<Scalar>();
    if (cfg["rl"]["body_rate_anneal_steps"]) body_rate_anneal_steps_ = cfg["rl"]["body_rate_anneal_steps"].as<Scalar>();
    if (cfg["rl"]["gate_width"]) gate_width_ = cfg["rl"]["gate_width"].as<Scalar>();
    if (cfg["rl"]["safety_coeff"]) safety_a_ = cfg["rl"]["safety_coeff"].as<Scalar>();
    if (cfg["rl"]["safety_d_max"]) safety_d_max_ = cfg["rl"]["safety_d_max"].as<Scalar>();
    // Distributed init params
    if (cfg["rl"]["traj_init_threshold"]) traj_init_threshold_ = cfg["rl"]["traj_init_threshold"].as<Scalar>();
    if (cfg["rl"]["traj_init_fraction"]) traj_init_fraction_ = cfg["rl"]["traj_init_fraction"].as<Scalar>();
    if (cfg["rl"]["traj_init_min_episodes"]) traj_init_min_episodes_ = cfg["rl"]["traj_init_min_episodes"].as<int>();
    if (cfg["rl"]["spline_deviation_ratio"]) spline_deviation_ratio_ = cfg["rl"]["spline_deviation_ratio"].as<Scalar>();
    if (cfg["rl"]["deviation_penalty_coeff"]) deviation_penalty_coeff_ = cfg["rl"]["deviation_penalty_coeff"].as<Scalar>();
  } else {
    logger_.error("Missing rl config section");
    return false;
  }

  // Build spline after gates and params are loaded
  if (num_gates_ > 0) {
    buildTrackSpline();
  }
  return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) { act = quad_act_; return true; }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

// ═══════════════════════════════════════════════════════════════════
// Cubic Hermite spline helpers
// ═══════════════════════════════════════════════════════════════════

// Evaluate cubic Hermite basis at t ∈ [0,1]
static inline Vector<3> hermite_eval(
    const Vector<3>& p0, const Vector<3>& m0,
    const Vector<3>& p1, const Vector<3>& m1,
    Scalar t) {
  Scalar t2 = t * t;
  Scalar t3 = t2 * t;
  Scalar h00 = Scalar(2)*t3 - Scalar(3)*t2 + Scalar(1);
  Scalar h10 = t3 - Scalar(2)*t2 + t;
  Scalar h01 = -Scalar(2)*t3 + Scalar(3)*t2;
  Scalar h11 = t3 - t2;
  return h00*p0 + h10*m0 + h01*p1 + h11*m1;
}

void QuadrotorEnv::buildTrackSpline() {
  // Build cubic Hermite spline segments through [start, gate0, ..., gateN].
  // At each gate, the tangent direction matches the gate normal (fly-through
  // direction), scaled by chord length for smooth C1 interpolation.
  // At the start position, tangent points toward gate 0.

  spline_segments_.clear();
  spline_segments_.resize(num_gates_);

  // Collect all control points: start + gates
  std::vector<Vector<3>> ctrl_pts;
  std::vector<Vector<3>> tangent_dirs;
  ctrl_pts.reserve(num_gates_ + 1);
  tangent_dirs.reserve(num_gates_ + 1);

  ctrl_pts.push_back(start_pos_);
  // Start tangent: direction toward first gate
  Vector<3> start_dir = (gates_[0] - start_pos_);
  if (start_dir.norm() > Scalar(1e-6)) start_dir.normalize();
  tangent_dirs.push_back(start_dir);

  for (int i = 0; i < num_gates_; ++i) {
    ctrl_pts.push_back(gates_[i]);
    tangent_dirs.push_back(gate_normals_[i]);  // unit vector, fly-through dir
  }

  // Global arc-length accumulator (across all segments)
  Scalar global_arc = Scalar(0);

  for (int seg_i = 0; seg_i < num_gates_; ++seg_i) {
    // Hermite segment from ctrl_pts[seg_i] to ctrl_pts[seg_i+1]
    const Vector<3>& p0 = ctrl_pts[seg_i];
    const Vector<3>& p1 = ctrl_pts[seg_i + 1];
    Scalar chord = (p1 - p0).norm();

    // Scale tangent vectors by chord length for proper Hermite parameterization
    Vector<3> m0 = tangent_dirs[seg_i] * chord;
    Vector<3> m1 = tangent_dirs[seg_i + 1] * chord;

    auto& seg = spline_segments_[seg_i];
    seg.s_start = global_arc;
    seg.samples.resize(kSplineSamplesPerSegment);
    seg.arc_lengths.resize(kSplineSamplesPerSegment);

    seg.samples[0] = p0;
    seg.arc_lengths[0] = global_arc;

    for (int j = 1; j < kSplineSamplesPerSegment; ++j) {
      Scalar t = Scalar(j) / Scalar(kSplineSamplesPerSegment - 1);
      seg.samples[j] = hermite_eval(p0, m0, p1, m1, t);
      global_arc += (seg.samples[j] - seg.samples[j - 1]).norm();
      seg.arc_lengths[j] = global_arc;
    }

    seg.s_end = global_arc;
  }

  logger_.info("Built spline progress tracker: " +
               std::to_string(num_gates_) + " segments, total arc length = " +
               std::to_string(global_arc) + " m, deviation_ratio = " +
               std::to_string(spline_deviation_ratio_));
}

Scalar QuadrotorEnv::spline_progress(const Vector<3>& pos, int target_gate_idx) {
  // Find the nearest point on the spline segment leading to target_gate_idx.
  // Returns: (arc_length_of_nearest - segment_start) - ratio * lateral_deviation
  //
  // The score is relative to the segment start (0 at previous gate/start),
  // so moving backward within the segment yields negative values.
  // Lateral deviation is penalized proportionally to spline_deviation_ratio_.

  if (target_gate_idx < 0 || target_gate_idx >= num_gates_) return Scalar(0);
  const auto& seg = spline_segments_[target_gate_idx];
  if (seg.samples.empty()) return Scalar(0);

  // Brute-force nearest sample (N=200 samples, very fast)
  Scalar best_dist_sq = std::numeric_limits<Scalar>::max();
  int best_idx = 0;
  for (int j = 0; j < static_cast<int>(seg.samples.size()); ++j) {
    Scalar d2 = (pos - seg.samples[j]).squaredNorm();
    if (d2 < best_dist_sq) {
      best_dist_sq = d2;
      best_idx = j;
    }
  }

  // Refine: project onto the line segment between best_idx-1 and best_idx+1
  // to get sub-sample accuracy
  Scalar best_arc = seg.arc_lengths[best_idx];
  Scalar best_dist = std::sqrt(best_dist_sq);

  // Try refining against the neighboring segments for better accuracy
  for (int offset = -1; offset <= 1; offset += 2) {
    int j2 = best_idx + offset;
    if (j2 < 0 || j2 >= static_cast<int>(seg.samples.size())) continue;

    Vector<3> a = seg.samples[best_idx];
    Vector<3> b = seg.samples[j2];
    Vector<3> ab = b - a;
    Scalar ab_len2 = ab.squaredNorm();
    if (ab_len2 < Scalar(1e-12)) continue;

    Scalar t = (pos - a).dot(ab) / ab_len2;
    t = std::max(Scalar(0), std::min(Scalar(1), t));

    Vector<3> proj = a + t * ab;
    Scalar d = (pos - proj).norm();
    if (d < best_dist) {
      best_dist = d;
      // Interpolate arc length
      best_arc = seg.arc_lengths[best_idx] +
                 t * (seg.arc_lengths[j2] - seg.arc_lengths[best_idx]);
    }
  }

  // Clamp arc length to segment bounds (prevents going before segment start)
  best_arc = std::max(best_arc, seg.s_start);
  best_arc = std::min(best_arc, seg.s_end);

  // Score: progress along path (relative to segment start) minus deviation penalty
  Scalar progress = (best_arc - seg.s_start) -
                    spline_deviation_ratio_ * best_dist;
  return progress;
}

Scalar QuadrotorEnv::spline_nearest_dist(const Vector<3>& pos, int target_gate_idx) {
  // Returns the nearest Euclidean distance from pos to the spline segment.
  if (target_gate_idx < 0 || target_gate_idx >= num_gates_) return Scalar(0);
  const auto& seg = spline_segments_[target_gate_idx];
  if (seg.samples.empty()) return Scalar(0);

  Scalar best_dist_sq = std::numeric_limits<Scalar>::max();
  for (int j = 0; j < static_cast<int>(seg.samples.size()); ++j) {
    Scalar d2 = (pos - seg.samples[j]).squaredNorm();
    if (d2 < best_dist_sq) best_dist_sq = d2;
  }
  return std::sqrt(best_dist_sq);
}

void QuadrotorEnv::updateExtraInfo() {
  extra_info_["r_progress"] = last_progress_reward_;
  extra_info_["r_safety"] = last_safety_reward_;
  extra_info_["r_bodyrate"] = last_body_rate_penalty_;
  extra_info_["r_deviation"] = last_deviation_penalty_;
  extra_info_["r_total"] = last_total_reward_;
  extra_info_["gate_idx"] = Scalar(current_gate_idx_);
  extra_info_["s_prev"] = s_prev_;
  extra_info_["pos_x"] = quad_state_.p(0);
  extra_info_["pos_y"] = quad_state_.p(1);
  extra_info_["pos_z"] = quad_state_.p(2);
  extra_info_["vel_norm"] = quad_state_.v.norm();
  extra_info_["gates_passed"] = Scalar(gates_passed_this_ep_);
  extra_info_["term_reason"] = Scalar(term_reason_);
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "num_gates =          [" << quad_env.num_gates_ << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
