#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace flightlib {

// Friend class for accessing private members
class QuadrotorEnvTestAccess {
 public:
  static std::string testCfgPath() {
    return std::string(getenv("FLIGHTMARE_PATH")) +
           "/flightlib/configs/test_env.yaml";
  }

  static QuadState& state(QuadrotorEnv& e) { return e.quad_state_; }
  static std::shared_ptr<Quadrotor>& quad(QuadrotorEnv& e) { return e.quadrotor_ptr_; }
  static int& gateIdx(QuadrotorEnv& e) { return e.current_gate_idx_; }
  static int numGates(QuadrotorEnv& e) { return e.num_gates_; }
  static Scalar& sPrev(QuadrotorEnv& e) { return e.s_prev_; }
  static bool passedGate(QuadrotorEnv& e) { return e.passed_gate_this_step_; }
  static int gatesPassedEp(QuadrotorEnv& e) { return e.gates_passed_this_ep_; }
  // is_gate_collision_ removed — gate collisions are disabled
  static Scalar terminalReward(QuadrotorEnv& e) { return e.terminal_reward_; }
  static Command& cmd(QuadrotorEnv& e) { return e.cmd_; }
  static int& totalEpisodes(QuadrotorEnv& e) { return e.total_episodes_; }
  static int& totalGatesPassed(QuadrotorEnv& e) { return e.total_gates_passed_; }
  static std::deque<TrajectorySnapshot>& trajBuffer(QuadrotorEnv& e) { return e.traj_buffer_; }
  static bool& useTrajInit(QuadrotorEnv& e) { return e.use_traj_init_; }
  static Scalar& trajInitFraction(QuadrotorEnv& e) { return e.traj_init_fraction_; }
  static int& trajBufferMax(QuadrotorEnv& e) { return e.traj_buffer_max_; }
  static int stepCounter(QuadrotorEnv& e) { return e.step_counter_; }
  static Vector<3>& prevVel(QuadrotorEnv& e) { return e.prev_vel_; }
  static std::vector<Vector<3>>& gates(QuadrotorEnv& e) { return e.gates_; }
  static std::vector<Vector<3>>& gateNormals(QuadrotorEnv& e) { return e.gate_normals_; }
  static Vector<3>& startPos(QuadrotorEnv& e) { return e.start_pos_; }
  static Vector<quadenv::kNAct>& actMean(QuadrotorEnv& e) { return e.act_mean_; }
  static Vector<quadenv::kNAct>& actStd(QuadrotorEnv& e) { return e.act_std_; }
  static Scalar gateWidth(QuadrotorEnv& e) { return e.gate_width_; }
  static long& totalSteps(QuadrotorEnv& e) { return e.total_steps_; }

  static Scalar splineProgress(QuadrotorEnv& e, const Vector<3>& pos, int idx) {
    return e.spline_progress(pos, idx);
  }
  static Scalar& deviationRatio(QuadrotorEnv& e) { return e.spline_deviation_ratio_; }
  static std::vector<SplineSegment>& splineSegs(QuadrotorEnv& e) { return e.spline_segments_; }

  // Teleport drone to a position (reset physics state)
  static void teleport(QuadrotorEnv& e, const Vector<3>& pos) {
    auto& s = e.quad_state_;
    s.x(QS::POSX) = pos(0);
    s.x(QS::POSY) = pos(1);
    s.x(QS::POSZ) = pos(2);
    e.quadrotor_ptr_->reset(s);
  }
};

}  // namespace flightlib

using namespace flightlib;
using TA = QuadrotorEnvTestAccess;

// ═══════════════════════════════════════════════════════════════
// TEST 3: Spline Progress Geometry
// ═══════════════════════════════════════════════════════════════

TEST(SplineProgress, OnAxisLinear) {
  // Straight-line track: spline is a straight line, so on-axis
  // progress matches arc length (= Euclidean distance from segment start).
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  // Set deviation ratio to 0 so we test pure arc-length progress
  TA::deviationRatio(env) = 0.0;

  // Segment 0: start(0,0,5) → gate0(20,0,5), arc_len ≈ 20
  EXPECT_NEAR(TA::splineProgress(env, Vector<3>(0,0,5), 0), 0.0, 0.5);
  EXPECT_NEAR(TA::splineProgress(env, Vector<3>(10,0,5), 0), 10.0, 0.5);
  EXPECT_NEAR(TA::splineProgress(env, Vector<3>(20,0,5), 0), 20.0, 0.5);
}

TEST(SplineProgress, BackwardYieldsZero) {
  // Moving behind the segment start should clamp to 0
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  TA::deviationRatio(env) = 0.0;

  // Behind start_pos → nearest point is sample[0], arc_len = s_start → progress = 0
  Scalar s = TA::splineProgress(env, Vector<3>(-5,0,5), 0);
  EXPECT_LE(s, 0.5);  // Should be at or near 0
}

TEST(SplineProgress, LateralDeviationPenalty) {
  // Lateral offset should reduce progress by deviation_ratio * distance
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  TA::deviationRatio(env) = 0.5;

  Scalar on_axis = TA::splineProgress(env, Vector<3>(10,0,5), 0);
  Scalar off_axis = TA::splineProgress(env, Vector<3>(10,5,5), 0);
  // off_axis should be reduced by approximately 0.5 * 5 = 2.5
  EXPECT_NEAR(on_axis - off_axis, 2.5, 0.5);
}

TEST(SplineProgress, Segment0UsesStartPos) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  TA::deviationRatio(env) = 0.0;

  // Segment 0 origin = start_pos (0,0,5), progress at (5,0,5) ≈ 5
  Scalar s = TA::splineProgress(env, Vector<3>(5,0,5), 0);
  EXPECT_NEAR(s, 5.0, 0.5);
}

TEST(SplineProgress, Segment1UsesPrevGate) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  TA::deviationRatio(env) = 0.0;

  // Segment 1: gates[0]=(20,0,5) → gates[1]=(40,0,5)
  // At (30,0,5), progress should be ≈ 10
  Scalar s = TA::splineProgress(env, Vector<3>(30,0,5), 1);
  EXPECT_NEAR(s, 10.0, 0.5);
}

// ═══════════════════════════════════════════════════════════════
// TEST 1: Reward Accumulation Along Straight Line
// ═══════════════════════════════════════════════════════════════

TEST(RewardAccumulation, StraightLineTotal) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  // drone at start_pos (0,0,5), gate_idx=0, target gate at (20,0,5)
  EXPECT_EQ(TA::gateIdx(env), 0);

  Scalar total_reward = 0.0;
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();
  Scalar delta = 0.5; // matches delta_max
  int steps = 40; // 40 * 0.5 = 20m = seg_len

  for (int i = 1; i <= steps; i++) {
    Vector<3> new_pos(i * delta, 0, 5);
    TA::teleport(env, new_pos);
    Scalar r = env.step(act, obs);
    total_reward += r;
  }
  // Paper Eq. 4: only progress (and a tiny body-rate penalty for stationary
  // hover, which is ~0). 40 steps * 0.5 m/step = 20 m segment length.
  EXPECT_NEAR(total_reward, 20.0, 0.5);
  EXPECT_EQ(TA::gateIdx(env), 1);
  EXPECT_EQ(TA::gatesPassedEp(env), 1);
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: Gate Pass and Collision Detection
// ═══════════════════════════════════════════════════════════════

TEST(GateDetection, SuccessfulPassThroughCenter) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Step 1: just before gate plane
  TA::teleport(env, Vector<3>(19.5, 0, 5));
  env.step(act, obs);
  EXPECT_FALSE(TA::passedGate(env));
  EXPECT_EQ(TA::gateIdx(env), 0);

  // Step 2: just past gate plane, through center (d_p ≈ 0)
  TA::teleport(env, Vector<3>(20.5, 0, 5));
  env.step(act, obs);
  EXPECT_TRUE(TA::passedGate(env));
  EXPECT_EQ(TA::gateIdx(env), 1);
  EXPECT_EQ(TA::gatesPassedEp(env), 1);
}

// Drone crosses gate plane outside the opening but within proximity.
// With gate collisions disabled, this should have NO effect —
// no termination, no gate advance, nothing.
TEST(GateDetection, OutsideOpeningNoEffect) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Place drone behind gate plane, laterally offset outside opening
  TA::teleport(env, Vector<3>(19.5, 2, 5));
  env.step(act, obs);

  // Cross the plane, still laterally offset
  TA::teleport(env, Vector<3>(20.5, 2, 5));
  env.step(act, obs);

  // No gate pass, no termination, still on gate 0
  EXPECT_FALSE(TA::passedGate(env));
  EXPECT_EQ(TA::gateIdx(env), 0);
  Scalar term_rew = 0;
  EXPECT_FALSE(env.isTerminalState(term_rew));
}

TEST(GateDetection, NoCrossingWhenFlyingWide) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Move laterally at s=10 (never crosses gate plane)
  for (int i = 0; i < 10; i++) {
    TA::teleport(env, Vector<3>(10, i * 0.5, 5));
    env.step(act, obs);
    EXPECT_FALSE(TA::passedGate(env));
  }
  EXPECT_EQ(TA::gateIdx(env), 0);
}

// Drone crosses the gate plane 30m away from the gate center.
// The proximity guard prevents this from doing anything.
TEST(GateDetection, DistantPlaneCrossingNoEffect) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Gate 0 at (20,0,5), normal [1,0,0]. Cross the plane at y=30 (far away).
  TA::teleport(env, Vector<3>(19.5, 30, 5));
  env.step(act, obs);

  TA::teleport(env, Vector<3>(20.5, 30, 5));
  env.step(act, obs);

  // Should NOT trigger anything — drone is 30m from gate center
  EXPECT_FALSE(TA::passedGate(env));
  EXPECT_EQ(TA::gateIdx(env), 0);
}

// Drone passes through the corner of the rectangular gate opening.
TEST(GateDetection, CornerPassIsSuccess) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Gate 0 at (20,0,5), normal [1,0,0], gate_width=1.5, half_w=0.75.
  // Pass through corner: offset of (0.7, 0.7) in gate plane
  TA::teleport(env, Vector<3>(19.5, 0.7, 5.7));
  env.step(act, obs);

  TA::teleport(env, Vector<3>(20.5, 0.7, 5.7));
  env.step(act, obs);

  EXPECT_TRUE(TA::passedGate(env));
  EXPECT_EQ(TA::gateIdx(env), 1);
}

TEST(GateDetection, MultiGateSequence) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  Vector<quadenv::kNAct> act = Vector<quadenv::kNAct>::Zero();

  // Pass through 3 gates sequentially
  for (int g = 0; g < 3; g++) {
    Scalar gate_x = (g + 1) * 20.0;
    TA::teleport(env, Vector<3>(gate_x - 0.5, 0, 5));
    env.step(act, obs);
    TA::teleport(env, Vector<3>(gate_x + 0.5, 0, 5));
    env.step(act, obs);
    EXPECT_TRUE(TA::passedGate(env));
  }
  EXPECT_EQ(TA::gateIdx(env), 0); // wraps around
  EXPECT_EQ(TA::gatesPassedEp(env), 3);
}

// ═══════════════════════════════════════════════════════════════
// TEST 5: Observation Space
// ═══════════════════════════════════════════════════════════════

TEST(ObservationSpace, DimensionLayout) {
  EXPECT_EQ(quadenv::kLinVel, 0);
  EXPECT_EQ(quadenv::kNLinVel, 3);
  EXPECT_EQ(quadenv::kLinAcc, 3);
  EXPECT_EQ(quadenv::kNLinAcc, 3);
  EXPECT_EQ(quadenv::kRotMat, 6);
  EXPECT_EQ(quadenv::kNRotMat, 9);
  EXPECT_EQ(quadenv::kAngVel, 15);
  EXPECT_EQ(quadenv::kNAngVel, 3);
  EXPECT_EQ(quadenv::kGateObs, 18);
  EXPECT_EQ(quadenv::kNPerGate, 4);
  EXPECT_EQ(quadenv::kNGateObs, 8);
  EXPECT_EQ(quadenv::kNObs, 26);
  EXPECT_EQ(quadenv::kNAct, 4);
}

TEST(ObservationSpace, IdentityRotationMatrix) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false); // deterministic: identity quat
  // Rotation matrix at identity should be I_3 flattened
  // obs[6:15] = [1,0,0, 0,1,0, 0,0,1]
  EXPECT_NEAR(obs(6), 1.0, 1e-4);  // R(0,0)
  EXPECT_NEAR(obs(7), 0.0, 1e-4);  // R(0,1)
  EXPECT_NEAR(obs(8), 0.0, 1e-4);  // R(0,2)
  EXPECT_NEAR(obs(9), 0.0, 1e-4);  // R(1,0)
  EXPECT_NEAR(obs(10), 1.0, 1e-4); // R(1,1)
  EXPECT_NEAR(obs(11), 0.0, 1e-4); // R(1,2)
  EXPECT_NEAR(obs(12), 0.0, 1e-4); // R(2,0)
  EXPECT_NEAR(obs(13), 0.0, 1e-4); // R(2,1)
  EXPECT_NEAR(obs(14), 1.0, 1e-4); // R(2,2)
}

TEST(ObservationSpace, GateDistanceReasonable) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  // Drone at (0,0,5), gate1 at (20,0,5) → r should be 20
  Scalar r1 = obs(quadenv::kGateObs); // obs[18] = distance to gate 1
  EXPECT_NEAR(r1, 20.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════
// TEST 6: Terminal State Detection
// ═══════════════════════════════════════════════════════════════

TEST(Terminal, GroundCrash) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  TA::teleport(env, Vector<3>(5, 0, 0.02));
  TA::quad(env)->reset(TA::state(env));
  Scalar r = 0;
  EXPECT_TRUE(env.isTerminalState(r));
  // Paper has no flat ground-crash penalty; episode just ends with reward 0.
  EXPECT_NEAR(r, 0.0, 1e-4);
}

TEST(Terminal, NotTerminalAboveGround) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  TA::teleport(env, Vector<3>(5, 0, 0.5));
  Scalar r = 0;
  EXPECT_FALSE(env.isTerminalState(r));
}

TEST(Terminal, OutOfBounds) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  TA::teleport(env, Vector<3>(0, 0, 101));
  Scalar r = 0;
  EXPECT_TRUE(env.isTerminalState(r));
}

TEST(Terminal, Timeout) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  TA::cmd(env).t = 30.0; // max_t
  Scalar r = 0;
  EXPECT_TRUE(env.isTerminalState(r));
  EXPECT_NEAR(r, 0.0, 1e-6);
}

TEST(Terminal, EpisodeCounterOnlyOnTerminal) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);

  int before = TA::totalEpisodes(env);
  Scalar r = 0;
  // Non-terminal calls should NOT increment
  for (int i = 0; i < 50; i++) {
    env.isTerminalState(r);
  }
  EXPECT_EQ(TA::totalEpisodes(env), before);

  // Terminal call should increment by exactly 1
  TA::teleport(env, Vector<3>(5, 0, 0.01));
  env.isTerminalState(r);
  EXPECT_EQ(TA::totalEpisodes(env), before + 1);
}

// ═══════════════════════════════════════════════════════════════
// TEST 8: Action Normalization
// ═══════════════════════════════════════════════════════════════

TEST(ActionNorm, HoverThrust) {
  QuadrotorEnv env(TA::testCfgPath());
  Scalar mass = 0.73;
  Scalar g = 9.81;
  Scalar expected_mean = mass * g / 4.0;
  Scalar expected_std = mass * 2.0 * g / 4.0;
  EXPECT_NEAR(TA::actMean(env)(0), expected_mean, 0.01);
  EXPECT_NEAR(TA::actStd(env)(0), expected_std, 0.01);
  // Zero action → hover thrust per motor
  // Total = 4 * act_mean ≈ mass * g
  EXPECT_NEAR(4.0 * TA::actMean(env)(0), mass * g, 0.05);
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Distributed Initialization
// ═══════════════════════════════════════════════════════════════

TEST(DistributedInit, Phase1WhenBufferEmpty) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  // Set conditions that WOULD trigger Phase 2 except buffer is empty
  TA::totalEpisodes(env) = 300;
  TA::totalGatesPassed(env) = 150;
  // Buffer is empty → should stay Phase 1
  for (int i = 0; i < 20; i++) {
    env.reset(obs, true);
  }
  EXPECT_FALSE(TA::useTrajInit(env));
}

TEST(DistributedInit, Phase2TransitionFires) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  TA::totalEpisodes(env) = 201;
  TA::totalGatesPassed(env) = 100;

  // Fill buffer with known snapshots
  for (int i = 0; i < 110; i++) {
    TrajectorySnapshot snap;
    snap.pos = Vector<3>(10, 10, 5);
    snap.vel = Vector<3>(5, 0, 0);
    snap.quat = Quaternion(1, 0, 0, 0);
    snap.omega = Vector<3>::Zero();
    snap.gate_idx = 0;
    TA::trajBuffer(env).push_back(snap);
  }

  env.reset(obs, true);
  EXPECT_TRUE(TA::useTrajInit(env));
}

TEST(DistributedInit, Phase2NeverBelowThreshold) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  TA::totalEpisodes(env) = 300;
  TA::totalGatesPassed(env) = 30; // rate = 0.1 < 0.3

  for (int i = 0; i < 200; i++) {
    TrajectorySnapshot snap;
    snap.pos = Vector<3>(10, 10, 5);
    snap.vel = Vector<3>(5, 0, 0);
    snap.quat = Quaternion(1, 0, 0, 0);
    snap.omega = Vector<3>::Zero();
    snap.gate_idx = 0;
    TA::trajBuffer(env).push_back(snap);
  }

  for (int i = 0; i < 20; i++) {
    env.reset(obs, true);
  }
  EXPECT_FALSE(TA::useTrajInit(env));
}

// ═══════════════════════════════════════════════════════════════
// TEST 9: Trajectory Snapshot Buffer
// ═══════════════════════════════════════════════════════════════

TEST(TrajBuffer, CircularEviction) {
  QuadrotorEnv env(TA::testCfgPath());
  Vector<quadenv::kNObs> obs; obs.setZero();
  env.reset(obs, false);
  TA::trajBufferMax(env) = 5;

  // Manually add 10 snapshots
  for (int i = 0; i < 10; i++) {
    TrajectorySnapshot snap;
    snap.pos = Vector<3>(i, 0, 5);
    snap.vel = Vector<3>(5, 0, 0);
    snap.quat = Quaternion(1, 0, 0, 0);
    snap.omega = Vector<3>::Zero();
    snap.gate_idx = 0;
    TA::trajBuffer(env).push_back(snap);
    if (static_cast<int>(TA::trajBuffer(env).size()) > TA::trajBufferMax(env)) {
      TA::trajBuffer(env).pop_front();
    }
  }
  EXPECT_EQ(static_cast<int>(TA::trajBuffer(env).size()), 5);
  // Oldest should be snap with pos.x=5
  EXPECT_NEAR(TA::trajBuffer(env).front().pos(0), 5.0, 1e-6);
}
