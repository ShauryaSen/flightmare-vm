#!/usr/bin/env python3
"""
IROS 2021 Gate Racing — PPO Training Script (Stable Baselines3)

Trains a quadrotor to race through gates using the IROS 2021 reward:
  R_t = progress + gate_bonus + survival - control_penalty

Observation space (26-dim, IROS 2021 Paper methodology.tex L173, L183-191):
  Proprio (18): [lin_vel(3), lin_accel(3), rot_matrix(9), ang_vel(3)]
  Gate 1 (4):   [r, θ, φ, α] — spherical in body frame (next gate)
  Gate 2 (4):   [r, θ, φ, α] — spherical in gate 1's frame (2nd future)
  - No absolute position (position-invariant policy)
  - Rotation matrix instead of Euler angles (no gimbal lock)
  - Spherical gate obs separates distance from direction (paper L187)
  - 2 future gates in observation (experiments.tex L48)
  - Z-score observation normalization via VecNormalize (paper L198-199)

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
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

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

    Also saves VecNormalize stats alongside the model for proper evaluation.
    """

    def __init__(self, save_path, log_freq=10, min_steps_before_save=300_000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_freq = log_freq
        self.best_mean_reward = -np.inf
        self.min_steps_before_save = min_steps_before_save
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

            # ── Log to SB3's logger so it shows in the standard table ──
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)
            self.logger.record("rollout/episodes", n_eps)

            # Save best model + VecNormalize stats. Skip until we have enough
            # steps so an early-killed run can't latch a junk reward as "best"
            # and overwrite a previously-trained good model.
            steps_so_far = self.num_timesteps
            if (mean_reward > self.best_mean_reward
                    and steps_so_far >= self.min_steps_before_save):
                self.best_mean_reward = mean_reward
                save_file = os.path.join(self.save_path, "best_model")
                self.model.save(save_file)
                # Save VecNormalize stats for evaluation
                env = self.training_env
                if isinstance(env, VecNormalize):
                    env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
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
    raw_env = make_env()

    # ── IROS 2021 Priority 4: Observation Normalization ──
    # Paper methodology.tex L198-199: "We apply input normalization using
    #   z-scoring by computing observation statistics"
    # Using SB3's VecNormalize for running z-score normalization.
    # norm_obs=True: normalize observations (z-score)
    # norm_reward=False: paper does not mention reward normalization
    # clip_obs=10.0: clip normalized obs to avoid extreme values
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        env = VecNormalize.load(args.load_vecnorm, raw_env)
        env.training = True
        env.norm_reward = False
        print(f"  VecNormalize: loaded running stats from {args.load_vecnorm}")
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print(f"  VecNormalize: fresh stats (no --load-vecnorm)")

    tb_log = args.tb_log or "saved/tb_logs/iros_gate_racing"

    if args.load_model and os.path.exists(args.load_model):
        # Curriculum transfer: same hyperparameters, same obs schema → just
        # reload weights and continue PPO. New env, new tb log dir.
        model = PPO.load(
            args.load_model, env=env, device="cpu", tensorboard_log=tb_log
        )
        # SB3 PPO.load preserves the original optimizer state; we don't reset
        # it because continued fine-tuning works fine here (plan §2 transfer).
        if args.learning_rate is not None:
            # Required when fine-tuning a BC-pretrained policy: a smaller LR
            # preserves the imitation prior; the default 3e-4 wipes it out.
            model.learning_rate = args.learning_rate
            model.lr_schedule = lambda _: args.learning_rate
            for g in model.policy.optimizer.param_groups:
                g["lr"] = args.learning_rate
            print(f"  LR override: {args.learning_rate}")
        # Allow ent_coef override on resume too — loaded models keep the value
        # baked in at original train time, but we may want more exploration on
        # a new (harder) track.
        model.ent_coef = args.ent_coef
        print(f"  ent_coef set to: {args.ent_coef}")
        print(f"  Loaded policy weights from {args.load_model}")
    else:
        # ── PPO with IROS 2021 hyperparameters ──
        model = PPO(
            "MlpPolicy",
            env,
            # Learning
            learning_rate=3e-4,
            n_steps=750,  # steps per env per update (200 envs × 750 = 150K) (Issue 6)
            batch_size=150000,  # full-batch updates
            n_epochs=10,  # PPO epochs per update
            # GAE
            gamma=0.99,
            gae_lambda=0.95,
            # PPO clipping
            clip_range=0.2,
            # Regularization
            ent_coef=args.ent_coef,  # entropy bonus for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Network architecture (2-layer MLP, same as paper Priority 7)
            policy_kwargs=dict(
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            verbose=1,
            tensorboard_log=tb_log,
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
    print(f"  Obs normalization: VecNormalize (z-score)")
    print(f"{'='*60}\n")

    # ── Callbacks ──
    save_dir = args.save_path or "saved/best_iros_gate_racing"
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Track training reward & save best model
    reward_callback = TrainingRewardCallback(
        save_path=save_dir,
        log_freq=10,  # print stats every 10 iterations
        verbose=1,
    )

    # 2. Save checkpoints every 500K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo",
    )

    callbacks = CallbackList([reward_callback, checkpoint_callback])

    # ── Train ──
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(args.load_model),
    )
    elapsed = time.time() - t0

    # Save final model + VecNormalize stats
    model.save(os.path.join(save_dir, "final_model"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"\n✓ Training complete in {elapsed/3600:.1f} hours")
    print(f"  Final model:      {save_dir}/final_model.zip")
    print(f"  Best model:       {save_dir}/best_model.zip")
    print(f"  VecNormalize:     {save_dir}/vec_normalize.pkl")


def evaluate(args):
    """Evaluate a trained model.

    Reports per-episode gates_passed (from C++ extra_info) for curriculum
    success criteria. Success is defined as `gates_passed >= args.min_gates`
    in `args.success_count` of `args.n_eval_episodes` episodes.
    """
    raw_env = make_env()

    model_path = args.model_path or "saved/best_iros_gate_racing/best_model.zip"
    vecnorm_path = args.vecnorm_path
    if vecnorm_path is None:
        # Default: look beside the model.
        guessed = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(guessed):
            vecnorm_path = guessed
        else:
            vecnorm_path = "saved/best_iros_gate_racing/vec_normalize.pkl"

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = False  # don't update stats during eval
        env.norm_reward = False
        print(f"  Loaded VecNormalize stats from: {vecnorm_path}")
    else:
        env = raw_env
        print(f"  WARNING: No VecNormalize stats found at {vecnorm_path}")
        print(f"  Running without observation normalization")

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Evaluation runs the full track from start_pos. Distributed init is a
    # training-only ingredient (paper L247-261); evaluating it would conflate
    # per-segment difficulty with end-to-end policy quality.
    raw_env.set_random_reset(False)

    n_episodes = args.n_eval_episodes
    max_steps = 200_000  # safety timeout

    episode_rewards = []
    episode_lengths = []
    episode_gates = []
    episode_term_reasons = []

    obs = env.reset()
    current_rewards = np.zeros(env.num_envs)
    current_lengths = np.zeros(env.num_envs, dtype=int)
    completed = 0
    total_steps = 0

    # We mark the first few envs as "active" so we collect deterministic eps
    # from the deterministic start_pos. (All envs start at start_pos because
    # set_random_reset(False), so any subset works.)
    active = np.zeros(env.num_envs, dtype=bool)
    active[: min(n_episodes, env.num_envs)] = True

    print(f"Running {n_episodes} deterministic evaluation episodes from start_pos...")
    while completed < n_episodes and total_steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_rewards += reward
        current_lengths += 1
        total_steps += 1

        for i in range(env.num_envs):
            if done[i] and active[i]:
                gp = float(info[i].get("gates_passed", 0.0))
                tr = float(info[i].get("term_reason", -1.0))
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                episode_gates.append(gp)
                episode_term_reasons.append(tr)
                completed += 1
                if completed % 5 == 0:
                    print(
                        f"  {completed}/{n_episodes} eps  "
                        f"last: gates={gp:.0f} len={current_lengths[i]} "
                        f"R={current_rewards[i]:.1f}"
                    )
                # Stop accumulating from this slot — we want N independent eps
                # not N×k where each slot keeps re-running.
                active[i] = False
            if done[i]:
                current_rewards[i] = 0
                current_lengths[i] = 0

        if not active.any() and completed < n_episodes:
            # Rare: exhaust active slots before reaching n_episodes (only if
            # n_eval_episodes > num_envs). Re-arm idle slots.
            need = n_episodes - completed
            for i in range(env.num_envs):
                if need <= 0:
                    break
                if not active[i]:
                    active[i] = True
                    need -= 1

    if completed < n_episodes:
        print(
            f"  WARNING: Only completed {completed}/{n_episodes} episodes "
            f"in {total_steps} steps (timeout)"
        )

    gates_arr = np.asarray(episode_gates)
    rewards_arr = np.asarray(episode_rewards)
    lengths_arr = np.asarray(episode_lengths)

    success_thresh = args.min_gates
    n_success = int(np.sum(gates_arr >= success_thresh))
    pass_criterion = n_success >= args.success_count

    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({completed} episodes, deterministic, from start_pos)")
    print(f"{'='*60}")
    print(f"  Mean gates_passed:    {gates_arr.mean():.2f} ± {gates_arr.std():.2f}")
    print(f"  Median gates_passed:  {np.median(gates_arr):.1f}")
    print(f"  Max gates_passed:     {int(gates_arr.max())}")
    print(f"  Min gates_passed:     {int(gates_arr.min())}")
    print(
        f"  Episodes with gates >= {success_thresh}: "
        f"{n_success}/{completed}  ({n_success/max(completed,1)*100:.0f}%)"
    )
    print(
        f"  Success criterion ({args.success_count}/{n_episodes} with "
        f"gates>={success_thresh}): {'✅ PASS' if pass_criterion else '❌ FAIL'}"
    )
    print(f"  Mean reward:          {rewards_arr.mean():.2f} ± {rewards_arr.std():.2f}")
    print(f"  Mean ep length:       {lengths_arr.mean():.0f} steps")
    print(f"{'='*60}\n")

    # Distribution: histogram of gates_passed counts.
    print("  Per-episode gates_passed distribution:")
    max_g = int(gates_arr.max())
    for g in range(max_g + 1):
        count = int(np.sum(gates_arr == g))
        bar = "█" * count
        print(f"    {g} gates : {count:>3}  {bar}")
    print()

    return {
        "n_episodes": completed,
        "mean_gates_passed": float(gates_arr.mean()),
        "n_success": n_success,
        "pass": bool(pass_criterion),
        "mean_reward": float(rewards_arr.mean()),
        "mean_length": float(lengths_arr.mean()),
    }


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
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl) for evaluation",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Path to a .zip to warm-start training (curriculum transfer).",
    )
    parser.add_argument(
        "--load-vecnorm", type=str, default=None,
        help="Path to VecNormalize .pkl to carry running stats across rungs.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override PPO LR after --load-model (recommended: 3e-5 when "
             "fine-tuning a BC-pretrained policy to preserve the IL prior).",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="PPO entropy coefficient (default 0.01). Bump to 0.02-0.05 if the "
             "policy collapses into a face-plant local optimum.",
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Directory to save best/final model + vecnormalize.pkl (default: saved/best_iros_gate_racing).",
    )
    parser.add_argument(
        "--tb-log", type=str, default=None,
        help="TensorBoard log dir override.",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=20,
        help="Number of deterministic eval episodes (default 20, matches curriculum plan).",
    )
    parser.add_argument(
        "--min-gates", type=int, default=1,
        help="Episode counts as a 'success' if gates_passed >= this value (default 1).",
    )
    parser.add_argument(
        "--success-count", type=int, default=18,
        help="Threshold #successful eps for the rung to count as cleared (default 18).",
    )
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
