#!/usr/bin/env python3
"""
Ablation Study Runner — Observation Space for Drone Gate Racing
===============================================================
Trains and evaluates PPO on a single ablation condition, saving
all artifacts to a named directory for later analysis.

Usage:
  # Train one condition
  python run_ablation.py --exp A_full --seed 0

  # List all available experiments
  python run_ablation.py --list

  # Evaluate an already-trained model
  python run_ablation.py --exp A_full --seed 0 --train 0

  # Quick smoke-test (1M steps)
  python run_ablation.py --exp A_full --seed 0 --total-timesteps 1000000
"""

import os
import sys
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.logger import configure

import flightgym
from sb3_vec_env_ablation import AblationVecEnv, ABLATIONS, AblationConfig


# ── Callback ──────────────────────────────────────────────────────────────────

class TrainingRewardCallback(BaseCallback):
    """Tracks episode reward/length from rollout buffer and saves best model.

    SB3's EvalCallback hangs with this C++ VecEnv (auto-reset on done means
    the eval loop never sees a terminal signal). This callback monitors the
    training reward instead.
    """

    def __init__(self, save_path: str, log_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_freq = log_freq
        self.best_mean_reward = -np.inf
        self.iteration = 0
        self._ep_rewards = np.zeros(0)
        self._ep_lengths = np.zeros(0, dtype=int)
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._initialized = False

    def _init_callback(self) -> None:
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=int)
        self._initialized = True

    def _on_step(self) -> bool:
        if not self._initialized:
            return True
        self._ep_rewards += self.locals["rewards"]
        self._ep_lengths += 1
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self._episode_rewards.append(self._ep_rewards[i])
                self._episode_lengths.append(self._ep_lengths[i])
                self._ep_rewards[i] = 0
                self._ep_lengths[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if len(self._episode_rewards) == 0:
            return

        mean_reward = np.mean(self._episode_rewards[-100:])
        mean_length = np.mean(self._episode_lengths[-100:])
        n_eps = len(self._episode_rewards)

        # Always log to TensorBoard
        self.logger.record("ablation/ep_rew_mean", mean_reward)
        self.logger.record("ablation/ep_len_mean", mean_length)
        self.logger.record("ablation/total_episodes", n_eps)

        if self.iteration % self.log_freq == 0 and self.verbose:
            print(f"\n  [Iter {self.iteration}] "
                  f"ep_rew_mean={mean_reward:.2f}, "
                  f"ep_len_mean={mean_length:.0f}, "
                  f"episodes={n_eps}, "
                  f"timesteps={self.num_timesteps:,}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            save_file = os.path.join(self.save_path, "best_model")
            self.model.save(save_file)
            if self.verbose:
                print(f"  ★ New best: {mean_reward:.2f} → {save_file}.zip")


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(ablation: AblationConfig, seed: int) -> AblationVecEnv:
    cpp_env = flightgym.QuadrotorEnv_v1()
    env = AblationVecEnv(cpp_env, ablation)
    env.seed(seed)
    return env


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, ablation: AblationConfig, run_dir: str) -> None:
    """Train PPO for one ablation condition + seed."""
    env = make_env(ablation, args.seed)

    tb_log_dir = os.path.join(run_dir, "tb_logs")
    os.makedirs(os.path.join(run_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=250,           # steps per env per update (100 envs × 250 = 25K)
        batch_size=25000,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        tensorboard_log=tb_log_dir,
        device="cpu",
        seed=args.seed,
    )

    print(f"\n{'='*65}")
    print(f"  Ablation Experiment: {ablation.name}")
    print(f"  {ablation.description}")
    print(f"  Seed: {args.seed}  |  Obs dim: {ablation.obs_dim}  |  "
          f"Steps: {args.total_timesteps:,}")
    print(f"  Output dir: {run_dir}")
    print(f"{'='*65}\n")

    reward_cb = TrainingRewardCallback(
        save_path=os.path.join(run_dir, "best_model"),
        log_freq=10,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=f"ablation_{ablation.name}_seed{args.seed}",
    )

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([reward_cb, checkpoint_cb]),
        progress_bar=True,
        tb_log_name=f"seed{args.seed}",
    )
    elapsed = time.time() - t0

    final_path = os.path.join(run_dir, f"final_seed{args.seed}")
    model.save(final_path)

    print(f"\n✓ Training complete in {elapsed / 60:.1f} min")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best model:  {run_dir}/best_model/best_model.zip")

    env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args, ablation: AblationConfig, run_dir: str) -> dict:
    """Evaluate a trained model. Returns a metrics dict."""
    env = make_env(ablation, args.seed)

    model_path = args.model_path or os.path.join(run_dir, "best_model", "best_model.zip")
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)

    n_episodes = args.eval_episodes
    episode_rewards, episode_lengths = [], []
    gate_counts = []  # number of gates passed per episode

    obs = env.reset()
    cur_rewards = np.zeros(env.num_envs)
    cur_lengths = np.zeros(env.num_envs, dtype=int)
    cur_gates = np.zeros(env.num_envs, dtype=int)
    completed = 0

    print(f"Running {n_episodes} evaluation episodes...")
    while completed < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cur_rewards += reward
        cur_lengths += 1

        # Count gate bonuses — a gate_bonus of 10.0 → each 10.0 chunk = 1 gate
        # We detect gate events by reward spikes > 9.0 in a single step
        for i in range(env.num_envs):
            if reward[i] > 9.0:  # gate bonus (10.0) dominates all other terms
                cur_gates[i] += 1

        for i in range(env.num_envs):
            if done[i]:
                episode_rewards.append(cur_rewards[i])
                episode_lengths.append(cur_lengths[i])
                gate_counts.append(cur_gates[i])
                cur_rewards[i] = 0
                cur_lengths[i] = 0
                cur_gates[i] = 0
                completed += 1
                if completed >= n_episodes:
                    break

    env.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    gate_counts = np.array(gate_counts)
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    metrics = {
        "exp": ablation.name,
        "seed": args.seed,
        "obs_dim": ablation.obs_dim,
        "n_episodes": completed,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "crash_rate": float(np.mean(episode_lengths < 490)),  # < 9.8s of 10s = crashed
        "gate_pass_rate_1": float(np.mean(gate_counts >= 1)),
        "gate_pass_rate_2": float(np.mean(gate_counts >= 2)),
        "gate_pass_rate_3": float(np.mean(gate_counts >= 3)),
        "mean_gates_passed": float(np.mean(gate_counts)),
    }

    print(f"\n{'='*65}")
    print(f"  Evaluation: {ablation.name}  (seed={args.seed})")
    print(f"{'='*65}")
    print(f"  Obs dim:         {ablation.obs_dim}")
    print(f"  Mean reward:     {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Max/Min reward:  {metrics['max_reward']:.2f} / {metrics['min_reward']:.2f}")
    print(f"  Mean ep length:  {metrics['mean_length']:.0f} steps")
    print(f"  Crash rate:      {metrics['crash_rate']*100:.1f}%")
    print(f"  Gates passed:    ≥1: {metrics['gate_pass_rate_1']*100:.1f}%  "
          f"≥2: {metrics['gate_pass_rate_2']*100:.1f}%  "
          f"≥3: {metrics['gate_pass_rate_3']*100:.1f}%")
    print(f"  Mean gates/ep:   {metrics['mean_gates_passed']:.2f}")
    print(f"{'='*65}\n")

    # Save metrics JSON
    import json
    metrics_path = os.path.join(run_dir, f"eval_metrics_seed{args.seed}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Observation ablation study runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Experiment name (e.g. A_full, B_no_euler, G2_noise_03). "
             "Use --list to see all options."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="1=train, 0=eval only")
    parser.add_argument(
        "--total-timesteps", type=int, default=25_000_000,
        help="Total training timesteps (default: 25M)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=500_000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model path for evaluation"
    )
    parser.add_argument(
        "--save-root", type=str, default="saved/ablations",
        help="Root directory for all ablation outputs"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiment names and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable ablation experiments:")
        print(f"  {'Name':<20} {'Obs dim':>8}  Description")
        print(f"  {'-'*20}  {'-'*8}  {'-'*45}")
        for name, cfg in ABLATIONS.items():
            print(f"  {name:<20} {cfg.obs_dim:>8}  {cfg.description[:60]}")
        print()
        sys.exit(0)

    if args.exp is None:
        parser.error("--exp is required (use --list to see options)")

    if args.exp not in ABLATIONS:
        print(f"ERROR: Unknown experiment '{args.exp}'")
        print(f"Available: {list(ABLATIONS.keys())}")
        sys.exit(1)

    ablation = ABLATIONS[args.exp]
    run_dir = os.path.join(args.save_root, f"{ablation.name}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    if args.train:
        train(args, ablation, run_dir)

    evaluate(args, ablation, run_dir)


if __name__ == "__main__":
    main()
