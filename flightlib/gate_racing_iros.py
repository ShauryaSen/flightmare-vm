#!/usr/bin/env python3
"""
IROS 2021 Gate Racing — PPO Training Script (Stable Baselines3)

Trains a quadrotor to race through gates using the IROS 2021 reward:
  R_t = progress + gate_bonus + survival - control_penalty

Observation space (21-dim, IROS 2021 Paper methodology.tex L173):
  [lin_vel(3), lin_accel(3), rot_matrix(9), ang_vel(3), rel_gate_body(3)]
  - No absolute position (position-invariant policy)
  - Rotation matrix instead of Euler angles (no gimbal lock)
  - Linear acceleration via finite difference

Action space (4-dim):
  [T1, T2, T3, T4] normalized per-motor thrusts

Usage:
  python gate_racing_iros.py --train 1 --total-timesteps 25000000
  python gate_racing_iros.py --train 0  # evaluate saved model
"""

import os
import sys
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

# Import the SB3-compatible VecEnv wrapper for flightgym
import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


class TrainingRewardCallback(BaseCallback):
    """
    Saves the best model based on training rollout reward.

    SB3's EvalCallback doesn't work with our custom C++ VecEnv because
    the auto-reset behavior confuses evaluate_policy()'s done-signal tracking.
    This callback monitors the training reward instead and saves checkpoints
    when a new best is reached.
    """

    def __init__(self, save_path, log_freq=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_freq = log_freq
        self.best_mean_reward = -np.inf
        self.iteration = 0
        # Track rewards ourselves from step data
        self._ep_rewards = np.zeros(0)
        self._ep_lengths = np.zeros(0, dtype=int)
        self._episode_rewards = []
        self._episode_lengths = []
        self._initialized = False

    def _init_callback(self) -> None:
        """Initialize per-env reward tracking."""
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=int)
        self._initialized = True

    def _on_step(self) -> bool:
        """Track per-step rewards to compute episode stats."""
        if not self._initialized:
            return True

        # Accumulate rewards and lengths
        self._ep_rewards += self.locals["rewards"]
        self._ep_lengths += 1

        # Check for episode completions
        dones = self.locals["dones"]
        for i in range(len(dones)):
            if dones[i]:
                self._episode_rewards.append(self._ep_rewards[i])
                self._episode_lengths.append(self._ep_lengths[i])
                self._ep_rewards[i] = 0
                self._ep_lengths[i] = 0

        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout collection (before PPO update)."""
        self.iteration += 1

        if len(self._episode_rewards) > 0:
            mean_reward = np.mean(self._episode_rewards[-100:])  # last 100 episodes
            mean_length = np.mean(self._episode_lengths[-100:])
            n_eps = len(self._episode_rewards)

            if self.iteration % self.log_freq == 0 and self.verbose:
                print(
                    f"\n  [Iter {self.iteration}] "
                    f"ep_rew_mean={mean_reward:.2f}, "
                    f"ep_len_mean={mean_length:.0f}, "
                    f"episodes={n_eps}, "
                    f"timesteps={self.num_timesteps:,}"
                )

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                save_file = os.path.join(self.save_path, "best_model")
                self.model.save(save_file)
                if self.verbose:
                    print(
                        f"  ★ New best reward: {mean_reward:.2f} → saved to {save_file}.zip"
                    )


def make_env():
    """Create the Flightmare vectorized environment."""
    cpp_env = flightgym.QuadrotorEnv_v1()
    env = FlightmareSB3VecEnv(cpp_env)
    print(f"Environment created:")
    print(f"  Obs dim:  {env.observation_space.shape}")
    print(f"  Act dim:  {env.action_space.shape}")
    print(f"  Num envs: {env.num_envs}")
    return env


def train(args):
    """Train PPO agent with IROS 2021 hyperparameters."""
    env = make_env()

    # ── PPO with IROS 2021 hyperparameters ──
    model = PPO(
        "MlpPolicy",
        env,
        # Learning
        learning_rate=3e-4,
        n_steps=250,  # steps per env per update (100 envs × 250 = 25K)
        batch_size=25000,  # full-batch updates
        n_epochs=10,  # PPO epochs per update
        # GAE
        gamma=0.99,
        gae_lambda=0.95,
        # PPO clipping
        clip_range=0.2,
        # Regularization
        ent_coef=0.05,  # entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Network architecture (2-layer MLP, same as paper)
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        verbose=1,
        tensorboard_log="saved/tb_logs/iros_gate_racing",
        device="cpu",  # ARM VM doesn't have GPU
    )

    print(f"\n{'='*60}")
    print(f"  IROS 2021 Gate Racing — PPO Training")
    print(f"{'='*60}")
    print(f"  Total timesteps:  {args.total_timesteps:,}")
    print(f"  n_steps:          {model.n_steps}")
    print(f"  batch_size:       {model.batch_size}")
    print(f"  n_epochs:         {model.n_epochs}")
    print(f"  learning_rate:    {model.learning_rate}")
    print(f"  gamma:            {model.gamma}")
    print(f"  ent_coef:         {model.ent_coef}")
    print(f"  Architecture:     {model.policy_kwargs}")
    print(f"{'='*60}\n")

    # ── Callbacks ──
    os.makedirs("saved/best_iros_gate_racing", exist_ok=True)
    os.makedirs("saved/checkpoints", exist_ok=True)

    # 1. Track training reward & save best model
    reward_callback = TrainingRewardCallback(
        save_path="saved/best_iros_gate_racing",
        log_freq=10,  # print stats every 10 iterations
        verbose=1,
    )

    # 2. Save checkpoints every 500K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path="saved/checkpoints",
        name_prefix="iros_gate_racing",
    )

    callbacks = CallbackList([reward_callback, checkpoint_callback])

    # ── Train ──
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # Save final model
    model.save("saved/iros_gate_racing_final")
    print(f"\n✓ Training complete in {elapsed/3600:.1f} hours")
    print(f"  Final model: saved/iros_gate_racing_final.zip")
    print(f"  Best model:  saved/best_iros_gate_racing/best_model.zip")


def evaluate(args):
    """Evaluate a trained model."""
    env = make_env()

    model_path = args.model_path or "saved/best_iros_gate_racing/best_model.zip"
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Run evaluation episodes
    n_episodes = 50
    max_steps = 100_000  # safety timeout
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    current_rewards = np.zeros(env.num_envs)
    current_lengths = np.zeros(env.num_envs, dtype=int)
    completed = 0
    total_steps = 0

    print(f"Running {n_episodes} evaluation episodes...")
    while completed < n_episodes and total_steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_rewards += reward
        current_lengths += 1
        total_steps += 1

        for i in range(env.num_envs):
            if done[i]:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                current_rewards[i] = 0
                current_lengths[i] = 0
                completed += 1
                if completed % 10 == 0:
                    print(f"  {completed}/{n_episodes} episodes completed")
                if completed >= n_episodes:
                    break

    if completed < n_episodes:
        print(
            f"  WARNING: Only completed {completed}/{n_episodes} episodes "
            f"in {total_steps} steps (timeout)"
        )

    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({completed} episodes)")
    print(f"{'='*60}")
    print(
        f"  Mean reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"  Mean length:  {np.mean(episode_lengths):.0f} steps")
    print(f"  Max reward:   {np.max(episode_rewards):.2f}")
    print(f"  Min reward:   {np.min(episode_rewards):.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="IROS 2021 Gate Racing RL")
    parser.add_argument("--train", type=int, default=1, help="1=train, 0=evaluate")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=25_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100_000, help="Evaluate every N timesteps"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=500_000,
        help="Save checkpoint every N timesteps",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model for evaluation"
    )
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
