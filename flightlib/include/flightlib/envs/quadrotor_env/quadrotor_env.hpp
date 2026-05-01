#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <deque>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // ══════════════════════════════════════════════════════════════
  // IROS 2021 Observation Space (Paper methodology.tex L173, L183-191;
  //   experiments.tex L48: "two future gates in the observation")
  //
  //   s_quad  = [v_WB(3), v̇_WB(3), R_WB(9), ω_B(3)]       = 18 dims
  //   s_track = [o1(3), α1, o2(3), α2]                       =  8 dims
  //   Total   = 26 dims
  //
  // Gate 1 (next): spherical (r,θ,φ) in body frame
  // Gate 2 (2nd):  spherical (r,θ,φ) in gate 1's frame (paper L189)
  // ══════════════════════════════════════════════════════════════
  kObs = 0,
  //
  // ── Drone proprioception (18 dims) ──
  kLinVel = 0,        // linear velocity v_WB (world frame)
  kNLinVel = 3,
  kLinAcc = 3,        // linear acceleration v̇_WB (world frame)
  kNLinAcc = 3,
  kRotMat = 6,        // flattened rotation matrix R_WB (row-major, 3x3)
  kNRotMat = 9,
  kAngVel = 15,       // angular velocity ω_B (body frame)
  kNAngVel = 3,
  // ── Gate observations (8 dims = 4 per gate × 2 gates) ──
  kGateObs = 18,      // index where gate obs starts
  kNPerGate = 4,      // (r, θ, φ, α) per gate
  kNFutureGates = 2,  // number of future gates in observation
  kNGateObs = 8,      // kNPerGate × kNFutureGates
  // ── Total ──
  kNObs = 26,         // 18 proprio + 8 gate
  // control actions (unchanged)
  kAct = 0,
  kNAct = 4,
};
};  // namespace quadenv

// ── Snapshot of a drone state for trajectory replay (Phase 2 init) ──
struct TrajectorySnapshot {
  Vector<3> pos;
  Vector<3> vel;
  Quaternion quat;
  Vector<3> omega;
  int gate_idx;  // which gate was the target when this state was recorded
};

// ── Precomputed spline segment for arc-length-based progress tracking ──
// Each segment spans from the previous gate (or start) to the next gate,
// storing densely-sampled (position, arc_length) pairs for fast nearest-
// point queries.
struct SplineSegment {
  std::vector<Vector<3>> samples;     // sampled 3D positions along spline
  std::vector<Scalar> arc_lengths;    // cumulative arc length at each sample
  Scalar s_start;                     // arc length at segment start (prev gate)
  Scalar s_end;                       // arc length at segment end (next gate)
};

class QuadrotorEnv final : public EnvBase {
  // Grant test classes access to private members for verification
  friend class QuadrotorEnvTestAccess;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void updateExtraInfo() override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  // ── Gate racing state (IROS 2021) ──
  std::vector<Vector<3>> gates_;         // gate center positions (world frame)
  std::vector<Vector<3>> gate_normals_;  // gate normal vectors (world frame, unit)
  std::vector<Matrix<3, 3>> gate_frames_; // full rotation matrices for gate frames
  Vector<3> start_pos_{Vector<3>::Zero()};  // start position from YAML
  int current_gate_idx_{0};              // index of next gate to pass
  int num_gates_{0};                     // total number of gates
  Scalar s_prev_{0.0};                  // projected progress at previous step (Eq. 2)
  Scalar gate_radius_{1.0};              // radius for gate-pass detection
  bool passed_gate_this_step_{false};    // flag for gate-pass bonus
  Vector<3> prev_pos_for_gate_detect_{Vector<3>::Zero()}; // position at previous step for gate-plane crossing

  // ── Distributed Initialization (Paper methodology.tex L247-261) ──
  // Phase 1: Hover init across all segments (uniform)
  // Phase 2: Sample from trajectory replay buffer (realistic velocities)
  // Transition: when gate-pass success rate exceeds threshold
  std::deque<TrajectorySnapshot> traj_buffer_;  // circular buffer of good states
  int traj_buffer_max_{5000};        // max snapshots stored
  int traj_snapshot_interval_{10};   // save state every N steps during episode
  int step_counter_{0};              // steps since last reset (for snapshot timing)
  int gates_passed_this_ep_{0};      // gates passed in current episode
  int total_episodes_{0};            // total episodes completed
  int total_gates_passed_{0};        // cumulative gates passed
  Scalar traj_init_threshold_{0.3};  // mean gates/episode required to switch to Phase 2
  int traj_init_min_episodes_{200};  // per-env episode floor before transition is even considered
  bool use_traj_init_{false};        // are we in Phase 2?
  Scalar traj_init_fraction_{0.5};   // fraction of resets using traj buffer in Phase 2
  long total_steps_{0};              // cumulative steps across all episodes

  // ── Acceleration tracking (for v̇_WB obs) ──
  Vector<3> prev_vel_{Vector<3>::Zero()};  // velocity at previous timestep

  // ── Reward coefficients (Paper Eq. 4-5) ──
  Scalar body_rate_coeff_{0.0};        // b in −b·‖ω‖² penalty
  Scalar body_rate_anneal_steps_{0};   // steps over which to anneal b to 0
  Scalar gate_width_{1.2};             // w_g (gate side length, AlphaPilot/Split-S)
  Scalar safety_a_{0.0};               // a in Eq. 4 (safety weight)
  Scalar safety_d_max_{1.0};           // d_max — threshold to activate safety reward
  bool is_gate_collision_{false};      // flag for gate collision terminal
  Scalar terminal_reward_{0.0};        // r_T (Eq. 5) computed on gate collision
  Scalar deviation_penalty_coeff_{0.01}; // per-step quadratic penalty beyond gate_radius

  // ── Reward component tracking (for extra_info logging) ──
  Scalar last_progress_reward_{0.0};
  Scalar last_safety_reward_{0.0};
  Scalar last_body_rate_penalty_{0.0};
  Scalar last_deviation_penalty_{0.0};
  Scalar last_total_reward_{0.0};
  int term_reason_{0};  // 0=running, 1=gate_collision, 2=ground, 3=oob, 4=timeout

  // ── Spline-based progress tracking ──
  std::vector<SplineSegment> spline_segments_;  // one per gate
  Scalar spline_deviation_ratio_{0.5};          // penalty per meter off spline
  static constexpr int kSplineSamplesPerSegment = 200;

  void buildTrackSpline();  // called from loadParam() after gates are loaded
  Scalar spline_progress(const Vector<3>& pos, int target_gate_idx);
  Scalar spline_nearest_dist(const Vector<3>& pos, int target_gate_idx);

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;

  // action normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib
