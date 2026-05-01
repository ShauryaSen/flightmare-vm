#!/usr/bin/env python3
"""Train + evaluate one sweep condition. Invoked as a subprocess.

Usage:
    python _train_one.py <run_dir> <total_timesteps> [--ent-coef X]
                         [--learning-rate X]

Reads the (already-patched) flightlib/configs/quadrotor_env.yaml at startup,
trains PPO for `total_timesteps`, then runs deterministic eval from start_pos
and writes result.json + progress.json to <run_dir>.

Run as subprocess so each condition gets a fresh Python interpreter and the
C++ env reloads its YAML cleanly.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch.nn as nn

# Make the parent training/ dir importable for sb3_vec_env
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


class ProgressCallback(BaseCallback):
    """Logs ep_rew_mean / ep_len_mean every `log_every` steps to JSON.

    Also tracks best mean reward and saves model + VecNormalize when beaten.
    """

    def __init__(self, run_dir, log_every=100_000, verbose=1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.log_every = log_every
        self.best_mean_reward = -np.inf
        self._ep_rewards = np.zeros(0)
        self._ep_lengths = np.zeros(0, dtype=int)
        self._ep_max_gates = np.zeros(0)  # max gate_idx seen this episode
        self._completed_rewards = []
        self._completed_lengths = []
        self._completed_max_gates = []
        self._initialized = False
        self._next_log = log_every
        self.progress = []  # list of {step, ep_rew_mean, ep_len_mean, ...}

    def _init_callback(self):
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=int)
        self._ep_max_gates = np.zeros(n)
        self._initialized = True

    def _on_step(self) -> bool:
        if not self._initialized:
            return True
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for i in range(len(dones)):
            gi = float(infos[i].get("gate_idx", 0.0))
            if gi > self._ep_max_gates[i]:
                self._ep_max_gates[i] = gi
            if dones[i]:
                self._completed_rewards.append(float(self._ep_rewards[i]))
                self._completed_lengths.append(int(self._ep_lengths[i]))
                self._completed_max_gates.append(float(self._ep_max_gates[i]))
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_max_gates[i] = 0.0

        if self.num_timesteps >= self._next_log:
            self._log_snapshot()
            self._next_log += self.log_every
        return True

    def _log_snapshot(self):
        if len(self._completed_rewards) == 0:
            return
        recent_n = min(100, len(self._completed_rewards))
        snap = {
            "step": int(self.num_timesteps),
            "ep_rew_mean": float(np.mean(self._completed_rewards[-recent_n:])),
            "ep_len_mean": float(np.mean(self._completed_lengths[-recent_n:])),
            "ep_max_gate_mean": float(np.mean(self._completed_max_gates[-recent_n:])),
            "ep_max_gate_max": float(np.max(self._completed_max_gates[-recent_n:])),
            "n_episodes": len(self._completed_rewards),
        }
        self.progress.append(snap)
        # Save the running curve every snapshot
        with open(os.path.join(self.run_dir, "progress.json"), "w") as f:
            json.dump(self.progress, f, indent=2)

        self.logger.record("rollout/ep_rew_mean", snap["ep_rew_mean"])
        self.logger.record("rollout/ep_len_mean", snap["ep_len_mean"])
        self.logger.record("rollout/ep_max_gate_mean", snap["ep_max_gate_mean"])

        if snap["ep_rew_mean"] > self.best_mean_reward:
            self.best_mean_reward = snap["ep_rew_mean"]
            best_dir = os.path.join(self.run_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save(os.path.join(best_dir, "best_model"))
            env = self.training_env
            if isinstance(env, VecNormalize):
                env.save(os.path.join(best_dir, "vec_normalize.pkl"))
            if self.verbose:
                print(f"  ★ step={snap['step']:>10,}  rew={snap['ep_rew_mean']:7.2f}  "
                      f"len={snap['ep_len_mean']:6.0f}  gate={snap['ep_max_gate_mean']:.2f}",
                      flush=True)


def evaluate(model, raw_env, vecnorm_env, n_episodes=10, max_steps=1500):
    """Deterministic eval from start_pos. Returns dict of metrics."""
    raw_env.set_random_reset(False)
    vecnorm_env.training = False
    vecnorm_env.norm_reward = False

    obs = vecnorm_env.reset()
    n_envs = vecnorm_env.num_envs
    cur_rewards = np.zeros(n_envs)
    cur_lengths = np.zeros(n_envs, dtype=int)
    cur_max_gates = np.zeros(n_envs)

    ep_rewards, ep_lengths, ep_max_gates = [], [], []
    completed = 0
    total_steps = 0

    while completed < n_episodes and total_steps < max_steps * 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vecnorm_env.step(action)
        cur_rewards += reward
        cur_lengths += 1
        for i in range(n_envs):
            gi = float(infos[i].get("gate_idx", 0.0))
            if gi > cur_max_gates[i]:
                cur_max_gates[i] = gi
            if done[i]:
                ep_rewards.append(float(cur_rewards[i]))
                ep_lengths.append(int(cur_lengths[i]))
                ep_max_gates.append(float(cur_max_gates[i]))
                cur_rewards[i] = 0.0
                cur_lengths[i] = 0
                cur_max_gates[i] = 0.0
                completed += 1
                if completed >= n_episodes:
                    break
        total_steps += 1

    if not ep_rewards:
        return {"eval_completed": 0, "eval_mean_reward": float("nan"),
                "eval_mean_length": 0, "eval_max_gate_idx": 0,
                "eval_lap_completed": False}

    max_gate_overall = max(ep_max_gates)
    return {
        "eval_completed": len(ep_rewards),
        "eval_mean_reward": float(np.mean(ep_rewards)),
        "eval_mean_length": float(np.mean(ep_lengths)),
        "eval_mean_max_gate": float(np.mean(ep_max_gates)),
        "eval_max_gate_idx": float(max_gate_overall),
        "eval_lap_completed": bool(max_gate_overall >= 10),
    }


def linfit_slope(progress, last_n_steps=2_000_000):
    """Return slope of ep_rew_mean over the final last_n_steps of progress log."""
    if len(progress) < 2:
        return 0.0
    end_step = progress[-1]["step"]
    start_step = end_step - last_n_steps
    pts = [(p["step"], p["ep_rew_mean"]) for p in progress if p["step"] >= start_step]
    if len(pts) < 2:
        return 0.0
    xs, ys = zip(*pts)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    # Slope per million steps for readability
    slope = np.polyfit(xs, ys, 1)[0] * 1_000_000
    return float(slope)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--total-timesteps", type=int, default=25_000_000)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    tb_dir = os.path.join(args.run_dir, "tb")

    # Build env + VecNormalize
    raw_env = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
    raw_env.seed(args.seed)
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        n_steps=1500,
        batch_size=150_000,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(activation_fn=nn.Tanh,
                           net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
        tensorboard_log=tb_dir,
        device="cpu",
        seed=args.seed,
    )

    cb = ProgressCallback(args.run_dir, log_every=args.log_every, verbose=1)

    t0 = time.time()
    print(f"[train] starting {args.run_dir}  steps={args.total_timesteps:,}  "
          f"ent_coef={args.ent_coef}  lr={args.learning_rate}", flush=True)

    train_error = None
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=cb,
                    progress_bar=False)
    except Exception as e:
        train_error = repr(e)
        print(f"[train] ERROR during learn(): {train_error}", flush=True)

    train_seconds = time.time() - t0

    # Save final model + VecNormalize
    model.save(os.path.join(args.run_dir, "final_model"))
    env.save(os.path.join(args.run_dir, "final_vec_normalize.pkl"))

    # Final progress dump (in case last snapshot wasn't on a log boundary)
    cb._log_snapshot()

    # ── Eval ─────────────────────────────────────────────────────────────
    eval_metrics = {}
    eval_error = None
    try:
        # Use best model for eval — recreate VecNormalize with saved stats
        best_dir = os.path.join(args.run_dir, "best_model")
        if os.path.exists(os.path.join(best_dir, "vec_normalize.pkl")):
            eval_raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
            eval_env = VecNormalize.load(
                os.path.join(best_dir, "vec_normalize.pkl"), eval_raw)
            best_model = PPO.load(os.path.join(best_dir, "best_model"))
            eval_metrics = evaluate(best_model, eval_raw, eval_env)
            eval_raw.close()
        else:
            eval_metrics = evaluate(model, raw_env, env)
    except Exception as e:
        eval_error = repr(e)
        print(f"[eval] ERROR: {eval_error}", flush=True)
        eval_metrics = {"eval_error": eval_error}

    # ── Final result.json ────────────────────────────────────────────────
    final_progress = cb.progress[-1] if cb.progress else {}
    result = {
        "run_dir": args.run_dir,
        "total_timesteps": args.total_timesteps,
        "train_seconds": train_seconds,
        "fps": (args.total_timesteps / train_seconds) if train_seconds > 0 else 0.0,
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_error": train_error,
        "final_ep_rew_mean": final_progress.get("ep_rew_mean", float("nan")),
        "final_ep_len_mean": final_progress.get("ep_len_mean", 0),
        "final_ep_max_gate_mean": final_progress.get("ep_max_gate_mean", 0),
        "best_ep_rew_mean": float(cb.best_mean_reward),
        "slope_last_2M": linfit_slope(cb.progress, 2_000_000),
        **eval_metrics,
    }
    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {args.run_dir}  rew={result['final_ep_rew_mean']:.2f}  "
          f"max_gate={result.get('eval_max_gate_idx', 'NA')}  "
          f"time={train_seconds/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
