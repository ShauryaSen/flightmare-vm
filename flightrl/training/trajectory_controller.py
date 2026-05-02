#!/usr/bin/env python3
"""
Trajectory Controller for Imitation Learning Warm-Start

Generates a smooth reference trajectory through gate centers on the circle_8 track,
then follows it using closed-loop PD control with inverse dynamics.
Collects (observation, action) pairs for behavioral cloning.

Usage:
  python trajectory_controller.py              # run controller, save demos + plot
  python trajectory_controller.py --plot-only   # just show the reference trajectory
"""

import os
import sys
import argparse
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─── Quadrotor Physical Parameters (loaded from quadrotor_env.yaml) ────────
_FLIGHTMARE_PATH = os.environ.get("FLIGHTMARE_PATH", os.path.expanduser("~/flightmare"))
_ENV_YAML = os.path.join(_FLIGHTMARE_PATH, "flightlib/configs/quadrotor_env.yaml")
with open(_ENV_YAML) as _f:
    _CFG = yaml.safe_load(_f)

_DYN = _CFG["quadrotor_dynamics"]
MASS = float(_DYN["mass"])
ARM_L = float(_DYN["arm_l"])
KAPPA = float(_DYN["kappa"])
G = 9.81                       # m/s² (Gz in flightlib is -9.81)
SIM_DT = float(_CFG["quadrotor_env"]["sim_dt"])
_INERTIA = _DYN["inertia"]
J = np.diag([float(_INERTIA[0]), float(_INERTIA[1]), float(_INERTIA[2])])
J_INV = np.linalg.inv(J)

# Motor allocation matrix t_BM (torque = t_BM @ thrusts)
# From dynamics code: arm_l * sqrt(0.5) * [[1,-1,-1,1],[-1,-1,1,1],[0,0,0,0]]
# Plus yaw: kappa * [1,-1,1,-1]
# Full 4x4 allocation: [total_thrust; roll_torque; pitch_torque; yaw_torque] = A @ [T1,T2,T3,T4]
_s = ARM_L * np.sqrt(0.5)
ALLOC = np.array([
    [1.0,     1.0,     1.0,     1.0    ],   # total thrust
    [_s,     -_s,     -_s,      _s     ],   # roll torque (x)
    [-_s,    -_s,      _s,      _s     ],   # pitch torque (y)
    [KAPPA,  -KAPPA,   KAPPA,  -KAPPA  ],   # yaw torque (z)
])
ALLOC_INV = np.linalg.inv(ALLOC)

# Action space mapping (from quadrotor_env.cpp lines 27-28):
#   act_mean = mass * g / 4  (per motor hover thrust)
#   act_std  = mass * 2 * g / 4  (range scaling)
#   physical_thrust = act * act_std + act_mean
#   → act = (physical_thrust - act_mean) / act_std
ACT_MEAN = MASS * G / 4.0   # ≈ 1.79 N per motor
ACT_STD  = MASS * 2 * G / 4.0  # ≈ 3.58 N per motor

# ─── Track Definition (loaded from YAML) ────────────────────────────────────
START_POS = np.array(_CFG["start_pos"], dtype=float)
GATE_POSITIONS = np.array(
    [g["position"] for g in _CFG["gates"]], dtype=float
)
N_GATES = len(GATE_POSITIONS)
# Track plane Z (used by ref-only plotter): take median gate z
Z_TRACK = float(np.median(GATE_POSITIONS[:, 2]))


def generate_reference_trajectory(speed=5.0, n_laps=1):
    """Generate a smooth trajectory through all gate centers.
    
    Uses cubic splines with periodic boundary conditions for the circular track.
    Returns arrays of (position, velocity, acceleration) sampled at SIM_DT.
    
    The trajectory starts from START_POS with zero velocity and ramps to cruise
    speed before reaching the first gate.
    
    Args:
        speed: Target average speed in m/s along the trajectory
        n_laps: Number of laps around the track
    """
    # Build waypoints: start → G0 → G1 → ... → G7 → G0 (close the loop for laps)
    waypoints = [START_POS.copy()]
    for lap in range(n_laps):
        for i in range(N_GATES):
            waypoints.append(GATE_POSITIONS[i].copy())
    # Add a final approach back near start
    waypoints.append(GATE_POSITIONS[0].copy())
    waypoints = np.array(waypoints)
    
    # Compute cumulative arc length as parameter
    dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(dists)])
    total_length = s[-1]
    
    # Fit cubic spline to each coordinate
    cs_x = CubicSpline(s, waypoints[:, 0], bc_type='clamped')
    cs_y = CubicSpline(s, waypoints[:, 1], bc_type='clamped')
    cs_z = CubicSpline(s, waypoints[:, 2], bc_type='clamped')
    
    # Use a time-based speed profile with smooth ramp-up/down
    # Ramp from 0 to 'speed' over the first 2 seconds
    total_time = total_length / speed + 2.0  # extra time for ramp
    n_samples = int(total_time / SIM_DT) + 1
    t_arr = np.linspace(0, total_time, n_samples)
    
    # Speed profile: smooth ramp-up using tanh
    ramp_time = 2.0  # seconds to reach cruise speed
    speed_profile = speed * np.tanh(t_arr / ramp_time * 2.0)
    
    # Integrate speed profile to get arc-length
    s_arr = np.cumsum(speed_profile) * SIM_DT
    s_arr = np.clip(s_arr, 0, total_length)  # Don't go beyond track
    
    # Evaluate position
    pos = np.column_stack([cs_x(s_arr), cs_y(s_arr), cs_z(s_arr)])
    
    # Velocity = dp/ds * ds/dt  where ds/dt = speed_profile
    vel = np.column_stack([cs_x(s_arr, 1) * speed_profile,
                           cs_y(s_arr, 1) * speed_profile,
                           cs_z(s_arr, 1) * speed_profile])
    
    # Acceleration: numerical differentiation of velocity for smoothness
    acc = np.zeros_like(vel)
    acc[1:] = np.diff(vel, axis=0) / SIM_DT
    # Cap feedforward acceleration to prevent extreme attitudes
    ACC_MAX = 5.0  # m/s²  (corresponds to ~27° tilt)
    acc = np.clip(acc, -ACC_MAX, ACC_MAX)
    
    return t_arr, pos, vel, acc


def desired_attitude_from_acceleration(acc_des, yaw=0.0):
    """Compute desired rotation matrix from desired acceleration vector.
    
    The body z-axis must align with the total force direction (thrust + gravity).
    
    Args:
        acc_des: Desired acceleration in world frame [ax, ay, az]
        yaw: Desired heading angle (radians)
    
    Returns:
        R_des: 3x3 rotation matrix (body → world)
        total_thrust: Total thrust magnitude (N)
    """
    # Total force = m * (a_des + g)  — the thrust vector in world frame
    force_des = MASS * (acc_des + np.array([0, 0, G]))
    total_thrust = np.linalg.norm(force_des)
    
    if total_thrust < 1e-6:
        return np.eye(3), MASS * G
    
    # Body z-axis = normalized force direction
    z_b = force_des / total_thrust
    
    # Body x-axis: project [cos(yaw), sin(yaw), 0] onto plane perpendicular to z_b
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b)
    if y_b_norm < 1e-6:
        # z_b is parallel to x_c, use alternative
        y_b = np.array([0, 1, 0])
    else:
        y_b = y_b / y_b_norm
    x_b = np.cross(y_b, z_b)
    
    R_des = np.column_stack([x_b, y_b, z_b])
    return R_des, total_thrust


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def compute_desired_yaw_along_trajectory(vel):
    """Compute heading angle from velocity direction (face forward)."""
    return np.arctan2(vel[1], vel[0])


def pd_attitude_controller(R_cur, R_des, omega_cur, Kp_att=15.0, Kd_att=5.0):
    """PD controller on SO(3) for attitude tracking.
    
    Returns desired body torques [τx, τy, τz].
    """
    # Attitude error: 0.5 * vee(R_des^T R_cur - R_cur^T R_des)
    R_err = R_des.T @ R_cur - R_cur.T @ R_des
    # Extract the vee map (skew-symmetric → vector)
    err = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
    
    # PD torque in body frame
    torque = -Kp_att * err - Kd_att * omega_cur
    return J @ torque  # Convert angular acceleration to torque


def thrusts_to_normalized_action(thrusts):
    """Convert physical per-motor thrusts (N) to normalized [-1, 1] actions.
    
    physical_thrust = act * act_std + act_mean
    → act = (physical_thrust - act_mean) / act_std
    """
    act = (thrusts - ACT_MEAN) / ACT_STD
    return np.clip(act, -1.0, 1.0)


def pd_position_controller(pos_cur, vel_cur, pos_des, vel_des, acc_ff,
                            Kp=3.0, Kd=2.5):
    """PD position controller with feedforward acceleration.
    
    Returns desired acceleration in world frame.
    """
    pos_err = pos_des - pos_cur
    vel_err = vel_des - vel_cur
    acc_des = acc_ff + Kp * pos_err + Kd * vel_err
    return acc_des


def run_in_environment(t_ref, pos_ref, vel_ref, acc_ref, 
                       save_path="demos/circle_8_demos.npz",
                       Kp=3.0, Kd=2.5, Kp_att=15.0, Kd_att=5.0):
    """Run the tracking controller in the actual Flightmare environment.
    
    Uses ground-truth position from extra_info (pos_x, pos_y, pos_z)
    for accurate closed-loop tracking.
    
    Returns the collected demonstrations and flight data.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    import flightgym
    from sb3_vec_env import FlightmareSB3VecEnv
    
    cpp_env = flightgym.QuadrotorEnv_v1()
    env = FlightmareSB3VecEnv(cpp_env)
    env.set_random_reset(False)  # Start from start_pos
    
    n_envs = env.num_envs
    obs = env.reset()
    
    n_steps = len(t_ref)
    
    all_obs = []
    all_acts = []
    all_rewards = []
    all_positions = []
    all_ref_positions = []
    all_velocities = []
    step_rewards = []
    gates_passed_log = []
    
    print(f"Running tracking controller for {n_steps} steps ({n_steps * SIM_DT:.1f}s)...")
    print(f"  Reference trajectory: {len(pos_ref)} points, speed ~{np.linalg.norm(vel_ref[1]):.1f} m/s")
    print(f"  PD gains: Kp={Kp}, Kd={Kd}, Kp_att={Kp_att}, Kd_att={Kd_att}")
    
    total_reward = 0.0
    done_count = 0
    pos_cur = START_POS.copy()  # Initial position until first step gives us ground truth
    ref_step = 0  # Track reference step independently (resets on episode done)
    
    for step_idx in range(n_steps):
        # Get reference state at this timestep
        ref_idx = min(ref_step, len(pos_ref)-1)
        p_des = pos_ref[ref_idx]
        v_des = vel_ref[ref_idx]
        a_ff  = acc_ref[ref_idx]
        
        # ── Extract current state from observation (env 0) ──
        o = obs[0]
        vel_cur = o[0:3]
        R_flat = o[6:15]
        R_cur = R_flat.reshape(3, 3)
        omega_cur = o[15:18]
        
        # ── PD Position Controller ──
        acc_des = pd_position_controller(pos_cur, vel_cur, p_des, v_des, a_ff,
                                          Kp=Kp, Kd=Kd)
        
        # ── Compute desired attitude and thrust ──
        yaw_des = compute_desired_yaw_along_trajectory(v_des)
        R_des, total_thrust = desired_attitude_from_acceleration(acc_des, yaw_des)
        
        # ── PD Attitude Controller ──
        body_torques = pd_attitude_controller(R_cur, R_des, omega_cur,
                                               Kp_att=Kp_att, Kd_att=Kd_att)
        
        # ── Allocate to per-motor thrusts ──
        wrench = np.array([total_thrust, body_torques[0], body_torques[1], body_torques[2]])
        motor_thrusts = ALLOC_INV @ wrench
        
        # Clamp to physical limits (each motor: 0 to ~10 N)
        motor_thrusts = np.clip(motor_thrusts, 0.0, 10.0)
        
        # ── Convert to normalized actions ──
        action = thrusts_to_normalized_action(motor_thrusts)
        
        # Broadcast same action to all envs
        actions = np.tile(action, (n_envs, 1)).astype(np.float32)
        
        # ── Step environment ──
        obs_new, rewards, dones, infos = env.step(actions)
        
        # ── Read ground-truth position from extra_info ──
        pos_cur = np.array([
            infos[0].get('pos_x', pos_cur[0]),
            infos[0].get('pos_y', pos_cur[1]),
            infos[0].get('pos_z', pos_cur[2])
        ])
        
        # Log env 0
        all_obs.append(obs[0].copy())
        all_acts.append(action.copy())
        all_rewards.append(rewards[0])
        all_positions.append(pos_cur.copy())
        all_ref_positions.append(p_des.copy())
        all_velocities.append(vel_cur.copy())
        step_rewards.append(rewards[0])
        total_reward += rewards[0]
        ref_step += 1
        
        # Print periodic status
        if step_idx % 200 == 0 and step_idx > 0:
            gate_idx = infos[0].get('gate_idx', -1)
            err = np.linalg.norm(pos_cur - p_des)
            print(f"  step={step_idx:5d} t={step_idx*SIM_DT:5.1f}s "
                  f"gate={gate_idx:.0f} err={err:.2f}m "
                  f"R_cum={total_reward:.1f} "
                  f"pos=[{pos_cur[0]:.1f},{pos_cur[1]:.1f},{pos_cur[2]:.1f}]")
        
        if dones[0]:
            done_count += 1
            gp = infos[0].get('gates_passed', 0)
            tr = infos[0].get('term_reason', -1)
            print(f"  *** Episode done at step {step_idx} ({step_idx*SIM_DT:.2f}s): "
                  f"gates_passed={gp:.0f}, term_reason={tr:.0f}, "
                  f"total_reward={total_reward:.1f}")
            gates_passed_log.append(gp)
            # After autoreset, position goes back to start_pos
            total_reward = 0.0
            ref_step = 0  # Restart reference trajectory
            pos_cur = START_POS.copy()
        
        obs = obs_new
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  Tracking Controller Results")
    print(f"{'='*60}")
    print(f"  Total steps:     {n_steps}")
    print(f"  Episodes done:   {done_count}")
    if gates_passed_log:
        print(f"  Gates passed:    {gates_passed_log}")
    print(f"  Cumulative R:    {sum(step_rewards):.1f}")
    print(f"{'='*60}")
    
    # Save demonstrations
    all_obs = np.array(all_obs, dtype=np.float32)
    all_acts = np.array(all_acts, dtype=np.float32)
    all_positions = np.array(all_positions)
    all_ref_positions = np.array(all_ref_positions)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "demos", exist_ok=True)
    np.savez(save_path,
             observations=all_obs,
             actions=all_acts,
             rewards=np.array(all_rewards),
             positions=all_positions,
             ref_positions=all_ref_positions)
    print(f"\n  Saved {len(all_obs)} demo transitions to {save_path}")
    
    return all_positions, all_ref_positions, np.array(step_rewards), all_velocities


def plot_trajectory(pos_ref, pos_actual=None, rewards=None, velocities=None):
    """Visualize reference trajectory and actual flight path."""
    fig = plt.figure(figsize=(18, 10))
    
    # ── Top-down view ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(pos_ref[:, 0], pos_ref[:, 1], 'g--', alpha=0.7, label='Reference')
    if pos_actual is not None:
        ax1.plot(pos_actual[:, 0], pos_actual[:, 1], 'b-', alpha=0.8, label='Actual')
    for i, gp in enumerate(GATE_POSITIONS):
        ax1.plot(gp[0], gp[1], 'rs', markersize=10)
        ax1.annotate(f'G{i}', (gp[0]+0.5, gp[1]+0.5), fontsize=8, color='red')
    ax1.plot(START_POS[0], START_POS[1], 'go', markersize=12, label='Start')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-Down View (XY)')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ── Side view ──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pos_ref[:, 0], pos_ref[:, 2], 'g--', alpha=0.7, label='Reference')
    if pos_actual is not None:
        ax2.plot(pos_actual[:, 0], pos_actual[:, 2], 'b-', alpha=0.8, label='Actual')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ── 3D view ──
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'g--', alpha=0.5, label='Reference')
    if pos_actual is not None:
        ax3.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], 'b-', alpha=0.8, label='Actual')
    for i, gp in enumerate(GATE_POSITIONS):
        ax3.scatter(*gp, c='red', s=50)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3D Trajectory')
    ax3.legend(fontsize=8)
    
    # ── Rewards ──
    if rewards is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        t = np.arange(len(rewards)) * SIM_DT
        ax4.plot(t, rewards, 'b-', alpha=0.7)
        ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Reward')
        ax4.set_title(f'Per-Step Reward (total={np.sum(rewards):.1f})')
        ax4.grid(True, alpha=0.3)
    
    # ── Tracking error ──
    if pos_actual is not None:
        ax5 = fig.add_subplot(2, 3, 5)
        n = min(len(pos_actual), len(pos_ref))
        errors = np.linalg.norm(pos_actual[:n] - pos_ref[:n], axis=1)
        t = np.arange(n) * SIM_DT
        ax5.plot(t, errors, 'r-', alpha=0.7)
        ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Error (m)')
        ax5.set_title(f'Tracking Error (mean={np.mean(errors):.2f}m)')
        ax5.grid(True, alpha=0.3)
    
    # ── Speed profile ──
    if velocities is not None:
        ax6 = fig.add_subplot(2, 3, 6)
        speeds = np.linalg.norm(velocities, axis=1)
        t = np.arange(len(speeds)) * SIM_DT
        ax6.plot(t, speeds, 'purple', alpha=0.7)
        ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Speed (m/s)')
        ax6.set_title(f'Speed Profile (mean={np.mean(speeds):.1f} m/s)')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Trajectory Controller — circle_8', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('trajectory_controller_results.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved to trajectory_controller_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot reference trajectory, no env simulation')
    parser.add_argument('--speed', type=float, default=5.0,
                        help='Target speed along trajectory (m/s)')
    parser.add_argument('--laps', type=int, default=1,
                        help='Number of laps around the track')
    parser.add_argument('--save-path', type=str, default='demos/circle_8_demos.npz',
                        help='Path to save demonstration data')
    parser.add_argument('--Kp', type=float, default=3.0, help='Position P gain')
    parser.add_argument('--Kd', type=float, default=2.5, help='Position D gain')
    parser.add_argument('--Kp-att', type=float, default=15.0, help='Attitude P gain')
    parser.add_argument('--Kd-att', type=float, default=5.0, help='Attitude D gain')
    args = parser.parse_args()
    
    print("Generating reference trajectory...")
    t_ref, pos_ref, vel_ref, acc_ref = generate_reference_trajectory(
        speed=args.speed, n_laps=args.laps
    )
    print(f"  {len(t_ref)} waypoints, {t_ref[-1]:.1f}s total, "
          f"speed={args.speed} m/s")
    
    if args.plot_only:
        plot_trajectory(pos_ref)
        return
    
    # Run in the actual environment
    pos_actual, pos_refs, rewards, velocities = run_in_environment(
        t_ref, pos_ref, vel_ref, acc_ref, save_path=args.save_path,
        Kp=args.Kp, Kd=args.Kd, Kp_att=args.Kp_att, Kd_att=args.Kd_att
    )
    
    # Plot results
    n = min(len(pos_actual), len(pos_ref))
    plot_trajectory(pos_ref[:n], pos_actual[:n], rewards, velocities)


if __name__ == '__main__':
    main()
