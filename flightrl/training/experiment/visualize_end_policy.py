#!/usr/bin/env python3
"""Render the trained Condition C policy on a panel of curriculum tracks.

For a given final checkpoint, samples M independent perturbed tracks at each
of K complexity levels, runs the deterministic policy on each, and overlays
the actual flown trajectory on the spline + gates. This is the visual
sanity check for "did the random-track curriculum produce a competent
generalist?"

The policy runs in the same C++ vec env it trained in (200 parallel envs),
but we apply the SAME perturbed track to every env so a single rollout
yields N copies of the deterministic trajectory — we just keep one. We
sequence M draws by re-applying gates between rollouts.

Usage:
    python visualize_end_policy.py \\
        --model runs/C_seed0/final_model.zip \\
        --vecnorm runs/C_seed0/final_vec_normalize.pkl \\
        --baseline-yaml /path/to/linear_8.yaml \\
        --complexities 0.0 0.3 0.7 \\
        --n-tracks 4 \\
        --out-dir runs/C_seed0/end_policy_viz
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(THIS_DIR))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

import flightgym  # noqa: E402
from sb3_vec_env import FlightmareSB3VecEnv  # noqa: E402

from track_generator import TrackGenerator  # noqa: E402
from visualize_random_tracks import (  # noqa: E402
    build_spline, gate_endpoints_xy, draw_track_2d, draw_track_3d,
    GATE_HALF_WIDTH, GATE_HEIGHT_HALF,
)

MAX_EPISODE_STEPS = 1500


def apply_to_all_envs(raw_env, positions, yaws):
    n = raw_env.num_envs
    p32 = positions.astype(np.float32)
    y32 = yaws.astype(np.float32)
    for i in range(n):
        raw_env.wrapper.applyPerturbedGates(i, p32, y32)


def rollout_one(model, raw_env, vec_env, max_steps: int = MAX_EPISODE_STEPS):
    """Run a single deterministic episode on env 0 (others share the track
    but we ignore them). Returns the per-step (x, y, z) trajectory plus the
    gates_passed count from extra_info."""
    raw_env.set_random_reset(False)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()
    n_envs = vec_env.num_envs
    done0 = False
    steps = 0
    traj = []
    final_gates = 0.0
    while not done0 and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec_env.step(action)
        # Sample env 0's quad position via getObs offsets? The 26-dim obs
        # has no absolute pos. Easier: use the spline-progress arc via
        # `gate_idx`/`gates_passed`. But we want the actual 3D trajectory.
        # The C++ env doesn't expose pos directly, so reconstruct via the
        # finite-difference of the lin-vel obs would be drift-prone.
        # Workaround: use raw VecEnv's testStep? That's the same env. We
        # simply step the env and read obs[0]; but obs has lin_vel, not pos.
        # Pragmatic alternative: re-create a dedicated single-env via testStep
        # — but that adds a pybind round-trip. For the visualization we use
        # an independent integrator below (see _integrate_traj).
        if done[0]:
            done0 = True
            final_gates = float(infos[0].get("gates_passed",
                                infos[0].get("gate_idx", 0.0)))
        steps += 1
    return final_gates, steps


def _integrate_traj_via_testStep(env_cpp, model, vec_env, raw_env,
                                 max_steps: int = MAX_EPISODE_STEPS):
    """Run a single deterministic rollout on env 0 and reconstruct (x,y,z)
    from finite-difference integration of the linear-velocity obs.

    This is the pragmatic path: the C++ env's 26-dim obs exposes v_WB[0:3]
    (world-frame velocity), so trajectory(t) ≈ start_pos + Σ v·dt. We use
    sim_dt=0.02s. Drift over a 30s episode is bounded by integration error
    on roughly-smooth velocity signals, which is fine for visualization.
    """
    raw_env.set_random_reset(False)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()
    # First-step velocity is the post-reset velocity.
    pos = np.zeros(3, dtype=np.float64)
    # Read the raw (un-normalized) obs from the wrapper to get true v_WB.
    raw_obs = vec_env.get_original_obs()  # SB3 inverse of normalize_obs
    pos[:] = 0.0  # We don't have absolute pos; reconstruct relative to start.

    sim_dt = 0.02
    traj = [pos.copy()]
    final_gates = 0.0
    n_envs = vec_env.num_envs
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec_env.step(action)
        raw_obs = vec_env.get_original_obs()
        v = raw_obs[0, 0:3].astype(np.float64)
        pos = pos + v * sim_dt
        traj.append(pos.copy())
        if done[0]:
            final_gates = float(infos[0].get("gates_passed",
                                infos[0].get("gate_idx", 0.0)))
            break
    return np.asarray(traj), final_gates


def render_panel(model, raw_env, vec_env, generator, complexities, n_tracks,
                 out_dir: Path):
    sp = generator.start_pos.astype(np.float64)
    n_complex = len(complexities)
    fig, axes = plt.subplots(n_complex, n_tracks,
                             figsize=(4.0 * n_tracks, 4.0 * n_complex),
                             squeeze=False)

    summary = []
    for ci, c in enumerate(complexities):
        for ti in range(n_tracks):
            ax = axes[ci][ti]
            pos, yaw = generator.sample(c)
            apply_to_all_envs(raw_env, pos, yaw)

            traj_rel, final_gates = _integrate_traj_via_testStep(
                raw_env.wrapper, model, vec_env, raw_env)
            # traj is reconstructed relative to start; lift back to world.
            traj = traj_rel + sp[None, :]
            summary.append({"c": c, "track": ti, "gates": final_gates,
                            "steps": int(len(traj_rel)) - 1})

            spline = build_spline(generator.start_pos, pos, yaw)
            ax.plot(spline[:, 0], spline[:, 1], color="goldenrod",
                    alpha=0.85, lw=1.8, label="ideal spline")
            for cg, y in zip(pos, yaw):
                (x0, y0), (x1, y1) = gate_endpoints_xy(cg, y)
                ax.plot([x0, x1], [y0, y1], color="goldenrod",
                        alpha=0.95, lw=2.0)
            ax.plot(traj[:, 0], traj[:, 1], color="navy", lw=1.5,
                    label=f"policy ({final_gates:.0f}/{generator.num_gates} gates)")
            ax.scatter([sp[0]], [sp[1]], marker="*", s=80, color="red",
                       zorder=10)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.set_title(f"c={c:.2f}, draw {ti+1} — "
                         f"{final_gates:.0f}/{generator.num_gates}",
                         fontsize=10)
            if ti == 0:
                ax.set_ylabel(f"complexity = {c:.2f}\n[m]")
            if ci == n_complex - 1:
                ax.set_xlabel("x [m]")
            ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    fig.suptitle("Condition C — final policy on curriculum tracks "
                 "(deterministic rollouts; navy = flown, gold = ideal spline)",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / "_end_policy_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--baseline-yaml",
                   default=str(THIS_DIR.parents[2] / "flightlib" / "configs"
                               / "tracks" / "linear_8.yaml"))
    p.add_argument("--complexities", nargs="+", type=float,
                   default=[0.0, 0.3, 0.7, 1.0])
    p.add_argument("--n-tracks", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
    raw.seed(args.seed)
    vec = VecNormalize.load(args.vecnorm, raw)
    vec.training = False
    vec.norm_reward = False
    raw.set_random_reset(False)

    model = PPO.load(args.model, device="cpu")
    print(f"loaded model: {args.model}")
    print(f"loaded vecnorm: {args.vecnorm}")

    gen = TrackGenerator.from_baseline_yaml(args.baseline_yaml,
                                            seed=args.seed)
    print(f"generator: {gen.num_gates} gates, forward={gen.forward}m, "
          f"max_lat={gen.lateral_max}, max_vert={gen.vertical_max}, "
          f"max_yaw={np.rad2deg(gen.yaw_max):.0f}°")

    summary = render_panel(model, raw, vec, gen, args.complexities,
                           args.n_tracks, out_dir)

    # Write a JSON summary too.
    import json
    with open(out_dir / "_end_policy_summary.json", "w") as f:
        json.dump({"complexities": args.complexities,
                   "n_tracks": args.n_tracks,
                   "results": summary}, f, indent=2)
    print(f"\nDone — see {out_dir}/_end_policy_grid.png")


if __name__ == "__main__":
    main()
