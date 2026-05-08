#!/usr/bin/env python3
"""Single (model x track [x perturbation]) evaluation subprocess.

Each invocation:
  - constructs a fresh QuadrotorEnv_v1() (reads quadrotor_env.yaml at ctor),
  - optionally calls applyPerturbedGates with a fixed-seed perturbation
    (robustness eval),
  - sets set_random_reset(False) and runs N deterministic episodes,
  - dumps per-episode gate_idx, ep_rew, ep_len arrays as JSON.

The orchestrator (run_eval_grid.py) is responsible for calling
track_loader.py apply <track> BEFORE invoking this — YAML mutation must be
inter-process.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
FLIGHTMARE_PATH = os.environ.get("FLIGHTMARE_PATH",
                                 str(THIS_DIR.parents[2]))
TRACKS_DIR = Path(FLIGHTMARE_PATH) / "flightlib" / "configs" / "tracks"

sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(THIS_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import flightgym
from sb3_vec_env import FlightmareSB3VecEnv



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--track", required=True,
                   help="Track name (must already be applied to quadrotor_env.yaml).")
    p.add_argument("--perturb-sigma", type=float, default=0.0,
                   help="If >0, apply a fixed Gaussian perturbation N(0, sigma^2) "
                        "to each baseline gate (robustness eval; per §7).")
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args()

    raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
    raw.seed(args.eval_seed)

    if args.perturb_sigma > 0.0:
        # §7 robustness eval: apply isotropic N(0, sigma^2) per-gate position
        # noise to the baseline track (independent of TrackGenerator's
        # curriculum-style sampling — this is the paper's "random gate
        # displacements" robustness check, not a curriculum draw).
        rng = np.random.default_rng(args.eval_seed)
        n_gates = raw.wrapper.getNumGates(0)
        baseline = np.zeros((n_gates, 3), dtype=np.float32)
        raw.wrapper.getGates(0, baseline)
        baseline_yaws = np.zeros(n_gates, dtype=np.float32)  # keep yaw=0 perturbation
        pos = baseline + rng.normal(0.0, args.perturb_sigma,
                                     size=baseline.shape).astype(np.float32)
        n_envs = raw.num_envs
        for i in range(n_envs):
            raw.wrapper.applyPerturbedGates(i,
                                            pos.astype(np.float32),
                                            baseline_yaws.astype(np.float32))

    env = VecNormalize.load(args.vecnorm, raw)
    env.training = False
    env.norm_reward = False
    raw.set_random_reset(False)

    model = PPO.load(args.model, device="cpu")
    obs = env.reset()
    n_envs = env.num_envs

    ep_rewards = np.zeros(n_envs)
    ep_lengths = np.zeros(n_envs, dtype=int)
    ep_max_gates = np.zeros(n_envs)
    completed_rewards, completed_lengths, completed_gates = [], [], []
    total_steps = 0
    while len(completed_rewards) < args.n_episodes and total_steps < args.max_steps * 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        ep_rewards += reward
        ep_lengths += 1
        for i in range(n_envs):
            gi = float(infos[i].get("gate_idx", 0.0))
            if gi > ep_max_gates[i]:
                ep_max_gates[i] = gi
            if done[i]:
                completed_rewards.append(float(ep_rewards[i]))
                completed_lengths.append(int(ep_lengths[i]))
                completed_gates.append(float(ep_max_gates[i]))
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                ep_max_gates[i] = 0.0
                if len(completed_rewards) >= args.n_episodes:
                    break
        total_steps += 1

    completed_rewards = completed_rewards[:args.n_episodes]
    completed_lengths = completed_lengths[:args.n_episodes]
    completed_gates = completed_gates[:args.n_episodes]

    out = {
        "model": args.model,
        "vecnorm": args.vecnorm,
        "track": args.track,
        "perturb_sigma": args.perturb_sigma,
        "eval_seed": args.eval_seed,
        "n_episodes": len(completed_rewards),
        "ep_rewards": completed_rewards,
        "ep_lengths": completed_lengths,
        "ep_max_gates": completed_gates,
        "mean_gates": float(np.mean(completed_gates)) if completed_gates else 0.0,
        "mean_reward": float(np.mean(completed_rewards)) if completed_rewards else 0.0,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_one] {args.track}  sigma={args.perturb_sigma}  "
          f"mean_gates={out['mean_gates']:.2f}  ({out['n_episodes']} eps)")
    raw.close()


if __name__ == "__main__":
    main()
