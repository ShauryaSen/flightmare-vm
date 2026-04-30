#!/usr/bin/env python3
"""
Visualize a trained gate-racing policy.

Runs evaluation episodes, records full trajectories (position, velocity,
orientation, reward), and generates:
  1. A static 3D trajectory plot (PNG) — all episodes overlaid
  2. An animated 3D video (GIF) — top-N episodes replayed in real-time
  3. A top-down (XY) trajectory plot (PNG) showing gate circuit
  4. Console summary with per-episode breakdown

Usage:
  python visualize_policy.py --model saved/best_iros_gate_racing/best_model.zip
  python visualize_policy.py --model saved/iros_gate_racing_final.zip -n 20
  python visualize_policy.py --model saved/iros_gate_racing_final.zip --animate
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
import ruamel.yaml

# Flightmare
import flightgym
from sb3_vec_env import FlightmareSB3VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


# ── Data structures ─────────────────────────────────────────────

@dataclass
class EpisodeTrajectory:
    """Recorded trajectory for one episode."""
    positions: np.ndarray      # (T, 3)
    velocities: np.ndarray     # (T, 3)
    orientations: np.ndarray   # (T, 3) euler ZYX
    actions: np.ndarray        # (T, 4)
    rewards: np.ndarray        # (T,)
    gate_rel: np.ndarray       # (T, 3) relative gate position in body frame
    total_reward: float = 0.0
    length: int = 0


# ── IROS 2021 Observation Space Layout (26-dim) ────────────────
# obs[0:3]   = linear velocity v_WB (world frame)
# obs[3:6]   = linear acceleration v̇_WB (world frame)
# obs[6:15]  = rotation matrix R_WB (flattened row-major, 3x3)
# obs[15:18] = angular velocity ω_B (body frame)
# obs[18:22] = gate 1: (r, θ, φ, α) spherical in body frame
# obs[22:26] = gate 2: (r, θ, φ, α) spherical in gate 1's frame
# NOTE: Position is NOT in the observation (position-invariant policy).
#       We reconstruct position by integrating velocity from start_pos.
OBS_VEL = slice(0, 3)       # linear velocity
OBS_ACC = slice(3, 6)       # linear acceleration
OBS_ROT = slice(6, 15)      # rotation matrix (9 values)
OBS_ANGVEL = slice(15, 18)  # angular velocity
OBS_GATE1 = slice(18, 22)   # gate 1 spherical obs (r, θ, φ, α)
OBS_GATE2 = slice(22, 26)   # gate 2 spherical obs
SIM_DT = 0.02               # physics timestep from config


# ── Environment + config ────────────────────────────────────────

def quat_to_R(w, x, y, z):
    """Quaternion to rotation matrix (matches C++ convention)."""
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def load_config(config_path: str):
    """Load gate positions, normals, and start position from the YAML config."""
    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)
    gates = []
    gate_normals = []
    for g in cfg.get("gates", []):
        gates.append(g["position"])
        if "rotation_quat" in g:
            q = g["rotation_quat"]
            R = quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            normal = R[:, 0]  # R * [1,0,0] — gate fly-through direction
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([1.0, 0.0, 0.0])
        gate_normals.append(normal)
    gates = np.array(gates, dtype=np.float32)
    gate_normals = np.array(gate_normals, dtype=np.float32)

    start_pos = np.array(cfg.get("start_pos", [0.0, 0.0, 2.0]),
                         dtype=np.float32)

    gate_width = 3.0  # default
    if "rl" in cfg and "gate_width" in cfg["rl"]:
        gate_width = float(cfg["rl"]["gate_width"])

    return gates, gate_normals, gate_width, start_pos


def make_env(vecnorm_path: str = None):
    """Create the Flightmare env, optionally wrapped with VecNormalize."""
    cpp_env = flightgym.QuadrotorEnv_v1()
    raw_env = FlightmareSB3VecEnv(cpp_env)

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = False   # don't update running stats during eval
        env.norm_reward = False
        print(f"  Loaded VecNormalize stats from: {vecnorm_path}")
        return env, raw_env
    else:
        if vecnorm_path:
            print(f"  WARNING: VecNormalize stats not found at {vecnorm_path}")
        print(f"  Running without observation normalization")
        return raw_env, raw_env


# ── Trajectory recording ────────────────────────────────────────

def get_raw_obs(env, raw_env):
    """Get the unnormalized observations from the underlying C++ env.

    When VecNormalize wraps the env, env.step() returns z-scored obs.
    We need the raw velocity for position integration.
    """
    # raw_env._observation holds the last raw obs from C++
    return raw_env._observation.copy()


def record_episodes(model, env, raw_env, start_pos: np.ndarray,
                    n_episodes: int, deterministic: bool = True
                    ) -> List[EpisodeTrajectory]:
    """Run the policy and record per-episode trajectories.

    Position is NOT in the observation space (position-invariant policy).
    We reconstruct position by integrating the raw velocity from the
    known start position: pos[t+1] = pos[t] + vel[t] * dt
    """
    obs = env.reset()
    n_envs = env.num_envs

    # Get raw (unnormalized) obs for velocity integration
    raw_obs = get_raw_obs(env, raw_env)

    # Per-env accumulators
    ep_positions = [[] for _ in range(n_envs)]
    ep_velocities = [[] for _ in range(n_envs)]
    ep_orientations = [[] for _ in range(n_envs)]
    ep_actions_list = [[] for _ in range(n_envs)]
    ep_rewards = [[] for _ in range(n_envs)]
    ep_gate_rel = [[] for _ in range(n_envs)]

    # Current integrated position for each env (start at start_pos)
    current_pos = np.tile(start_pos, (n_envs, 1)).astype(np.float32)

    completed: List[EpisodeTrajectory] = []
    max_steps = 200_000  # safety

    for step in range(max_steps):
        # Record current state
        for i in range(n_envs):
            ep_positions[i].append(current_pos[i].copy())
            ep_velocities[i].append(raw_obs[i, OBS_VEL].copy())
            # Extract rotation matrix (3x3 from 9 flattened values)
            R_flat = raw_obs[i, OBS_ROT]
            ep_orientations[i].append(R_flat[:3].copy())  # first row of R
            # Gate 1 spherical obs: (r, θ, φ, α)
            ep_gate_rel[i].append(raw_obs[i, OBS_GATE1][:3].copy())

        # Step (obs is normalized if VecNormalize is active — correct for model)
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        # Get raw obs for velocity integration
        raw_obs = get_raw_obs(env, raw_env)

        # Integrate velocity to update position: pos += vel * dt
        for i in range(n_envs):
            ep_actions_list[i].append(action[i].copy())
            ep_rewards[i].append(reward[i])
            current_pos[i] += raw_obs[i, OBS_VEL] * SIM_DT

        # Check for completed episodes
        for i in range(n_envs):
            if done[i]:
                traj = EpisodeTrajectory(
                    positions=np.array(ep_positions[i]),
                    velocities=np.array(ep_velocities[i]),
                    orientations=np.array(ep_orientations[i]),
                    actions=np.array(ep_actions_list[i]),
                    rewards=np.array(ep_rewards[i]),
                    gate_rel=np.array(ep_gate_rel[i]),
                    total_reward=sum(ep_rewards[i]),
                    length=len(ep_rewards[i]),
                )
                completed.append(traj)

                # Reset accumulators and position for this env
                ep_positions[i] = []
                ep_velocities[i] = []
                ep_orientations[i] = []
                ep_actions_list[i] = []
                ep_rewards[i] = []
                ep_gate_rel[i] = []
                current_pos[i] = start_pos.copy()  # reset to start

                if len(completed) >= n_episodes:
                    break

        if len(completed) >= n_episodes:
            break

        if (step + 1) % 500 == 0 and len(completed) < n_episodes:
            print(f"  ... {len(completed)}/{n_episodes} episodes "
                  f"({step+1} steps)")

    return completed


# ── Static plots ─────────────────────────────────────────────────

def plot_3d_trajectories(episodes: List[EpisodeTrajectory],
                         gates: np.ndarray,
                         gate_normals: np.ndarray,
                         gate_width: float,
                         start_pos: np.ndarray,
                         out_path: str,
                         title: str = "Policy Trajectories",
                         top_n: int = None):
    """Plot 3D trajectories with gate markers."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Sort by reward (best first)
    sorted_eps = sorted(episodes, key=lambda e: e.total_reward, reverse=True)
    if top_n:
        sorted_eps = sorted_eps[:top_n]

    # Color map: best=green, worst=red
    n = len(sorted_eps)
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for idx, ep in enumerate(sorted_eps):
        pos = ep.positions
        alpha = 0.9 if idx < 3 else 0.3
        lw = 2.0 if idx < 3 else 0.8
        label = (f"R={ep.total_reward:.1f} ({ep.length} steps)"
                 if idx < 5 else None)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=colors[idx], alpha=alpha, linewidth=lw, label=label)

        # Mark start
        ax.scatter(*pos[0], color=colors[idx], marker="o", s=30, alpha=0.8)
        # Mark end (crash/timeout)
        ax.scatter(*pos[-1], color=colors[idx], marker="x", s=40, alpha=0.8)

    # Draw gates with normals, segment directions, and openings
    for i, g in enumerate(gates):
        # Gate opening ring perpendicular to the gate normal
        normal = gate_normals[i]
        # Build a frame perpendicular to the normal
        up = np.array([0, 0, 1.0])
        if abs(np.dot(normal, up)) > 0.95:
            up = np.array([0, 1.0, 0])
        perp1 = np.cross(normal, up)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        theta = np.linspace(0, 2 * np.pi, 50)
        r = gate_width / 2  # gate opening radius
        gx = g[0] + r * (np.cos(theta)[:, None] * perp1 + np.sin(theta)[:, None] * perp2)[:, 0]
        gy = g[1] + r * (np.cos(theta)[:, None] * perp1 + np.sin(theta)[:, None] * perp2)[:, 1]
        gz = g[2] + r * (np.cos(theta)[:, None] * perp1 + np.sin(theta)[:, None] * perp2)[:, 2]
        ax.plot(gx, gy, gz, "b-", linewidth=2.5, alpha=0.8)
        ax.scatter(*g, color="blue", s=100, marker="D", zorder=5)

        # Gate NORMAL arrow (RED) — fly-through direction
        arrow_len = 3.0
        ax.quiver(g[0], g[1], g[2],
                  normal[0]*arrow_len, normal[1]*arrow_len, normal[2]*arrow_len,
                  color='red', linewidth=2.5, arrow_length_ratio=0.15)

        # SEGMENT direction arrow (CYAN) — approach from previous gate
        if i == 0:
            g_prev = start_pos
        else:
            g_prev = gates[i - 1]
        seg_dir = g - g_prev
        seg_dir = seg_dir / np.linalg.norm(seg_dir)
        ax.quiver(g[0], g[1], g[2],
                  seg_dir[0]*arrow_len*0.7, seg_dir[1]*arrow_len*0.7, seg_dir[2]*arrow_len*0.7,
                  color='cyan', linewidth=1.8, arrow_length_ratio=0.12, alpha=0.7)

        # Angle between normal and segment
        dot_val = np.dot(normal, seg_dir)
        angle = np.degrees(np.arccos(np.clip(abs(dot_val), 0, 1)))
        color = 'lime' if angle < 20 else ('yellow' if angle < 45 else 'red')
        ax.text(g[0], g[1], g[2] + 0.8, f"G{i} ({angle:.0f}°)", fontsize=9,
                ha="center", color=color, fontweight="bold")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if n <= 10:
        ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio
    all_pos = np.vstack([ep.positions for ep in sorted_eps])
    all_pts = np.vstack([all_pos, gates])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(max(0, center[2] - max_range), center[2] + max_range)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved 3D trajectory plot: {out_path}")


def plot_topdown(episodes: List[EpisodeTrajectory],
                 gates: np.ndarray,
                 gate_normals: np.ndarray,
                 gate_width: float,
                 start_pos: np.ndarray,
                 out_path: str,
                 title: str = "Top-Down View"):
    """Plot XY (top-down) and XZ (side) views."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sorted_eps = sorted(episodes, key=lambda e: e.total_reward, reverse=True)
    n = len(sorted_eps)
    cmap = plt.cm.RdYlGn

    for ax_idx, (ax, proj_label, dim_a, dim_b, xlabel, ylabel) in enumerate([
        (axes[0], "Top-down (XY)", 0, 1, "X (m)", "Y (m)"),
        (axes[1], "Side view (XZ)", 0, 2, "X (m)", "Z (m)"),
    ]):
        for idx, ep in enumerate(sorted_eps):
            pos = ep.positions
            color = cmap(idx / max(n - 1, 1))
            alpha = 0.8 if idx < 3 else 0.2
            lw = 2.0 if idx < 3 else 0.6
            ax.plot(pos[:, dim_a], pos[:, dim_b],
                    color=color, alpha=alpha, linewidth=lw)
            ax.scatter(pos[0, dim_a], pos[0, dim_b],
                       color=color, marker="o", s=20, alpha=0.7, zorder=3)
            ax.scatter(pos[-1, dim_a], pos[-1, dim_b],
                       color=color, marker="x", s=25, alpha=0.7, zorder=3)

        # Gates with normals and openings
        for i, g in enumerate(gates):
            normal = gate_normals[i]
            # Gate opening bar (perpendicular to normal, projected)
            gate_perp_2d = np.array([-normal[dim_b], normal[dim_a]])
            if np.linalg.norm(gate_perp_2d) > 1e-6:
                gate_perp_2d = gate_perp_2d / np.linalg.norm(gate_perp_2d)
            hw = gate_width / 2
            p1 = np.array([g[dim_a], g[dim_b]]) + gate_perp_2d * hw
            p2 = np.array([g[dim_a], g[dim_b]]) - gate_perp_2d * hw
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color='blue', linewidth=3, alpha=0.8, zorder=4)
            ax.plot(g[dim_a], g[dim_b], "bD", markersize=8, zorder=5)

            # Gate NORMAL arrow (RED)
            arrow_len = 3.0
            ax.annotate('', xy=(g[dim_a] + normal[dim_a]*arrow_len,
                                g[dim_b] + normal[dim_b]*arrow_len),
                        xytext=(g[dim_a], g[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                       lw=2.0, mutation_scale=15), zorder=6)

            # SEGMENT direction arrow (CYAN)
            if i == 0:
                g_prev_pos = start_pos
            else:
                g_prev_pos = gates[i - 1]
            seg_dir = g - g_prev_pos
            seg_norm = np.linalg.norm(seg_dir)
            if seg_norm > 1e-6:
                seg_dir = seg_dir / seg_norm
            ax.annotate('', xy=(g[dim_a] + seg_dir[dim_a]*arrow_len*0.7,
                                g[dim_b] + seg_dir[dim_b]*arrow_len*0.7),
                        xytext=(g[dim_a], g[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='cyan',
                                       lw=1.5, mutation_scale=12, alpha=0.7), zorder=5)

            # Angle label
            dot_val = np.dot(normal[:3], seg_dir[:3])
            angle = np.degrees(np.arccos(np.clip(abs(dot_val), 0, 1)))
            label_color = 'lime' if angle < 20 else ('yellow' if angle < 45 else 'red')
            ax.annotate(f"G{i} ({angle:.0f}°)", (g[dim_a], g[dim_b]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=label_color, fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(proj_label, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved top-down plot: {out_path}")


def plot_episode_detail(ep: EpisodeTrajectory, gates: np.ndarray,
                        gate_normals: np.ndarray, gate_width: float,
                        out_path: str, title: str = "Episode Detail"):
    """Detailed multi-panel view of a single episode."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    t = np.arange(ep.length) * 0.02  # sim_dt = 0.02s

    # 1. 3D trajectory
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    pos = ep.positions
    # Color by timestep
    for j in range(len(pos) - 1):
        frac = j / max(len(pos) - 1, 1)
        ax3d.plot(pos[j:j+2, 0], pos[j:j+2, 1], pos[j:j+2, 2],
                  color=plt.cm.viridis(frac), linewidth=1.5)
    ax3d.scatter(*pos[0], color="green", s=80, marker="o", label="Start")
    ax3d.scatter(*pos[-1], color="red", s=80, marker="x", label="End")
    for i, g in enumerate(gates):
        normal = gate_normals[i]
        # Gate opening perpendicular to normal
        up = np.array([0, 0, 1.0])
        if abs(np.dot(normal, up)) > 0.95:
            up = np.array([0, 1.0, 0])
        perp1 = np.cross(normal, up)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        theta_ring = np.linspace(0, 2 * np.pi, 40)
        hw = gate_width / 2
        rx = g[0] + hw * (np.cos(theta_ring)[:, None] * perp1 + np.sin(theta_ring)[:, None] * perp2)[:, 0]
        ry = g[1] + hw * (np.cos(theta_ring)[:, None] * perp1 + np.sin(theta_ring)[:, None] * perp2)[:, 1]
        rz = g[2] + hw * (np.cos(theta_ring)[:, None] * perp1 + np.sin(theta_ring)[:, None] * perp2)[:, 2]
        ax3d.plot(rx, ry, rz, "b-", linewidth=2, alpha=0.7)
        # Normal arrow
        ax3d.quiver(g[0], g[1], g[2],
                    normal[0]*2, normal[1]*2, normal[2]*2,
                    color='red', linewidth=2, arrow_length_ratio=0.2)
        ax3d.text(g[0], g[1], g[2] + 0.5, f"G{i}", color="blue",
                  fontsize=9, ha="center")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D Trajectory (color = time)")
    ax3d.legend(fontsize=8)

    # 2. Position over time
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.plot(t, pos[:, 0], label="x", color="#e74c3c")
    ax_pos.plot(t, pos[:, 1], label="y", color="#2ecc71")
    ax_pos.plot(t, pos[:, 2], label="z", color="#3498db")
    ax_pos.set_xlabel("Time (s)"); ax_pos.set_ylabel("Position (m)")
    ax_pos.set_title("Position vs Time")
    ax_pos.legend(); ax_pos.grid(True, alpha=0.3)

    # 3. Speed over time
    ax_vel = fig.add_subplot(gs[0, 2])
    speed = np.linalg.norm(ep.velocities, axis=1)
    ax_vel.plot(t, speed, color="#9b59b6", linewidth=1.5)
    ax_vel.fill_between(t, 0, speed, alpha=0.2, color="#9b59b6")
    ax_vel.set_xlabel("Time (s)"); ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.set_title("Speed vs Time")
    ax_vel.grid(True, alpha=0.3)

    # 4. Per-step reward
    ax_rew = fig.add_subplot(gs[1, 0])
    colors_rew = ["#2ecc71" if r > 0 else "#e74c3c" for r in ep.rewards]
    ax_rew.bar(t, ep.rewards, width=0.02, color=colors_rew, alpha=0.7)
    ax_rew.axhline(0, color="gray", linewidth=0.8)
    ax_rew.set_xlabel("Time (s)"); ax_rew.set_ylabel("Reward")
    ax_rew.set_title(f"Per-Step Reward (total={ep.total_reward:.1f})")
    ax_rew.grid(True, alpha=0.3)

    # 5. Distance to target gate
    ax_dist = fig.add_subplot(gs[1, 1])
    gate_dist = np.linalg.norm(ep.gate_rel, axis=1)
    ax_dist.plot(t, gate_dist, color="#e67e22", linewidth=1.5)
    ax_dist.axhline(1.0, color="blue", linestyle="--", alpha=0.5,
                    label="Gate radius")
    ax_dist.set_xlabel("Time (s)"); ax_dist.set_ylabel("Distance (m)")
    ax_dist.set_title("Distance to Target Gate")
    ax_dist.legend(); ax_dist.grid(True, alpha=0.3)

    # 6. Motor commands
    ax_act = fig.add_subplot(gs[1, 2])
    motor_labels = ["T1", "T2", "T3", "T4"]
    motor_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for j in range(4):
        ax_act.plot(t[:len(ep.actions)], ep.actions[:, j],
                    label=motor_labels[j], color=motor_colors[j],
                    alpha=0.7, linewidth=1)
    ax_act.set_xlabel("Time (s)"); ax_act.set_ylabel("Normalized Thrust")
    ax_act.set_title("Motor Commands")
    ax_act.legend(fontsize=8); ax_act.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved episode detail: {out_path}")


# ── Animation ────────────────────────────────────────────────────

def animate_episode(ep: EpisodeTrajectory, gates: np.ndarray,
                    out_path: str, speedup: int = 2):
    """Create an animated GIF of a single episode."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    pos = ep.positions
    all_pts = np.vstack([pos, gates])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3

    # Draw gates
    for i, g in enumerate(gates):
        theta = np.linspace(0, 2 * np.pi, 50)
        ax.plot(g[0] + np.cos(theta), g[1] + np.sin(theta),
                np.full(50, g[2]), "b-", linewidth=2.5, alpha=0.7)
        ax.scatter(*g, color="blue", s=100, marker="D", zorder=5)
        ax.text(g[0], g[1], g[2] + 0.5, f"G{i}", fontsize=10,
                ha="center", color="blue", fontweight="bold")

    # Trail line + drone marker
    trail_line, = ax.plot([], [], [], "r-", linewidth=1.5, alpha=0.6)
    drone_marker, = ax.plot([], [], [], "ro", markersize=8)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                          fontsize=11, fontweight="bold",
                          bbox=dict(boxstyle="round", facecolor="wheat",
                                    alpha=0.8))
    reward_text = ax.text2D(0.02, 0.88, "", transform=ax.transAxes,
                            fontsize=10,
                            bbox=dict(boxstyle="round", facecolor="lightyellow",
                                      alpha=0.8))

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(max(0, center[2] - max_range), center[2] + max_range)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Flight Animation (R={ep.total_reward:.1f})",
                 fontsize=13, fontweight="bold")

    # Select frames (subsample for speed)
    frame_indices = list(range(0, len(pos), speedup))
    if frame_indices[-1] != len(pos) - 1:
        frame_indices.append(len(pos) - 1)

    def update(frame_num):
        idx = frame_indices[frame_num]
        trail_line.set_data(pos[:idx+1, 0], pos[:idx+1, 1])
        trail_line.set_3d_properties(pos[:idx+1, 2])
        drone_marker.set_data([pos[idx, 0]], [pos[idx, 1]])
        drone_marker.set_3d_properties([pos[idx, 2]])
        t_sec = idx * 0.02
        time_text.set_text(f"t = {t_sec:.2f}s  (step {idx})")
        cum_rew = ep.rewards[:idx+1].sum()
        reward_text.set_text(f"Cumul. reward: {cum_rew:.1f}")
        return trail_line, drone_marker, time_text, reward_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=40, blit=False,
    )

    # Save as GIF
    writer = animation.PillowWriter(fps=25)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  ✓ Saved animation: {out_path}")


# ── Console summary ─────────────────────────────────────────────

def print_summary(episodes: List[EpisodeTrajectory]):
    """Print a table of episode stats."""
    rewards = [e.total_reward for e in episodes]
    lengths = [e.length for e in episodes]
    speeds = [np.linalg.norm(e.velocities, axis=1).mean() for e in episodes]
    max_speeds = [np.linalg.norm(e.velocities, axis=1).max() for e in episodes]

    print(f"\n{'='*65}")
    print(f"  Evaluation Summary — {len(episodes)} episodes")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Reward':<25} {np.mean(rewards):>10.2f} "
          f"{np.std(rewards):>10.2f} {np.min(rewards):>10.2f} "
          f"{np.max(rewards):>10.2f}")
    print(f"  {'Episode length (steps)':<25} {np.mean(lengths):>10.0f} "
          f"{np.std(lengths):>10.0f} {np.min(lengths):>10d} "
          f"{np.max(lengths):>10d}")
    print(f"  {'Avg speed (m/s)':<25} {np.mean(speeds):>10.2f} "
          f"{np.std(speeds):>10.2f} {np.min(speeds):>10.2f} "
          f"{np.max(speeds):>10.2f}")
    print(f"  {'Max speed (m/s)':<25} {np.mean(max_speeds):>10.2f} "
          f"{np.std(max_speeds):>10.2f} {np.min(max_speeds):>10.2f} "
          f"{np.max(max_speeds):>10.2f}")
    sim_times = [e.length * 0.02 for e in episodes]
    print(f"  {'Flight time (s)':<25} {np.mean(sim_times):>10.2f} "
          f"{np.std(sim_times):>10.2f} {np.min(sim_times):>10.2f} "
          f"{np.max(sim_times):>10.2f}")
    print(f"{'='*65}")

    # Top/bottom episodes
    sorted_eps = sorted(enumerate(episodes),
                        key=lambda x: x[1].total_reward, reverse=True)
    print(f"\n  Top 5 episodes:")
    for rank, (i, ep) in enumerate(sorted_eps[:5]):
        avg_spd = np.linalg.norm(ep.velocities, axis=1).mean()
        print(f"    #{rank+1}: R={ep.total_reward:7.2f}  "
              f"len={ep.length:4d}  "
              f"avg_speed={avg_spd:.2f} m/s  "
              f"flight_time={ep.length*0.02:.1f}s")

    print(f"\n  Bottom 3 episodes:")
    for rank, (i, ep) in enumerate(sorted_eps[-3:]):
        avg_spd = np.linalg.norm(ep.velocities, axis=1).mean()
        print(f"    R={ep.total_reward:7.2f}  "
              f"len={ep.length:4d}  "
              f"avg_speed={avg_spd:.2f} m/s")
    print()


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained gate-racing policy")
    parser.add_argument("--model", type=str,
                        default="saved/best_iros_gate_racing/best_model.zip",
                        help="Path to model .zip file")
    parser.add_argument("-n", "--n-episodes", type=int, default=20,
                        help="Number of evaluation episodes to record")
    parser.add_argument("--animate", action="store_true",
                        help="Create animated GIFs of top episodes")
    parser.add_argument("--animate-n", type=int, default=3,
                        help="Number of episodes to animate (top-N by reward)")
    parser.add_argument("--out-dir", type=str, default="viz",
                        help="Output directory for plots/animations")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (non-deterministic) policy")
    parser.add_argument("--speedup", type=int, default=2,
                        help="Animation speedup factor (1=real-time, 2=2x, ...)")
    parser.add_argument("--vecnorm", type=str,
                        default="saved/best_iros_gate_racing/vec_normalize.pkl",
                        help="Path to VecNormalize stats (.pkl) for proper eval")
    args = parser.parse_args()

    # Setup output directory
    model_name = Path(args.model).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{model_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gate positions and start position from config
    flightmare_path = os.environ.get("FLIGHTMARE_PATH",
                                     os.path.expanduser("~/flightmare"))
    config_path = os.path.join(flightmare_path,
                               "flightlib/configs/quadrotor_env.yaml")
    gates, gate_normals, gate_width, start_pos = load_config(config_path)
    print(f"Loaded {len(gates)} gates from config")
    print(f"Start position: {start_pos}")
    print(f"Gate width: {gate_width}m (opening radius: {gate_width/2}m)")

    # Create environment with VecNormalize for proper model inference
    env, raw_env = make_env(vecnorm_path=args.vecnorm)
    print(f"Environment: {env.num_envs} envs, "
          f"obs_dim={env.observation_space.shape}, "
          f"act_dim={env.action_space.shape}")

    # Load model
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # Record episodes
    # Position is reconstructed by integrating raw velocity from start_pos
    print(f"\nRecording {args.n_episodes} episodes "
          f"({'stochastic' if args.stochastic else 'deterministic'})...")
    print(f"  (Position reconstructed via velocity integration from start_pos)")
    episodes = record_episodes(
        model, env, raw_env, start_pos, args.n_episodes,
        deterministic=not args.stochastic,
    )
    print(f"Recorded {len(episodes)} episodes")

    # Summary
    print_summary(episodes)



    # 1. 3D trajectory overlay
    plot_3d_trajectories(
        episodes, gates, gate_normals, gate_width, start_pos,
        out_path=str(out_dir / f"{model_name}_3d_trajectories.png"),
        title=f"3D Trajectories — {model_name}",
    )

    # 2. Top-down + side view
    plot_topdown(
        episodes, gates, gate_normals, gate_width, start_pos,
        out_path=str(out_dir / f"{model_name}_topdown.png"),
        title=f"Trajectory Views — {model_name}",
    )

    # 3. Detailed view of best and worst episodes
    sorted_eps = sorted(episodes, key=lambda e: e.total_reward, reverse=True)
    plot_episode_detail(
        sorted_eps[0], gates, gate_normals, gate_width,
        out_path=str(out_dir / f"{model_name}_best_episode.png"),
        title=f"Best Episode (R={sorted_eps[0].total_reward:.1f}) — {model_name}",
    )
    plot_episode_detail(
        sorted_eps[-1], gates, gate_normals, gate_width,
        out_path=str(out_dir / f"{model_name}_worst_episode.png"),
        title=f"Worst Episode (R={sorted_eps[-1].total_reward:.1f}) — {model_name}",
    )

    # 4. Animations (optional)
    if args.animate:
        print(f"\nCreating {args.animate_n} animations...")
        for idx, ep in enumerate(sorted_eps[:args.animate_n]):
            out_path = str(out_dir / f"{model_name}_ep{idx+1}_R{ep.total_reward:.0f}.gif")
            animate_episode(ep, gates, out_path, speedup=args.speedup)

    # Save raw trajectory data for later analysis
    np.savez_compressed(
        str(out_dir / f"{model_name}_trajectories.npz"),
        rewards=[e.total_reward for e in episodes],
        lengths=[e.length for e in episodes],
        gates=gates,
        # Save best episode fully for reference
        best_positions=sorted_eps[0].positions,
        best_velocities=sorted_eps[0].velocities,
        best_rewards=sorted_eps[0].rewards,
    )
    print(f"  ✓ Saved trajectory data: {out_dir}/{model_name}_trajectories.npz")

    print(f"\n✓ All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
