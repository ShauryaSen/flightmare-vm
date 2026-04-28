#!/usr/bin/env python3
"""
Reward Probe — Diagnose the reward function step by step.

This script runs the environment with simple policies (hover, random, toward-gate)
and logs detailed reward statistics to verify:
  1. Reward is positive when drone moves toward gate
  2. Reward magnitude is reasonable
  3. Dones/terminals aren't firing too aggressively
  4. Progress increases monotonically for toward-gate policy

Usage:
  python reward_probe.py
"""

import os
import sys
import numpy as np

# Ensure flightgym is importable
sys.path.insert(0, os.path.dirname(__file__))
import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


def run_probe(policy_name, action_fn, n_steps=300, n_envs_to_show=3):
    """Run environment with a given action function and log rewards."""
    print(f"\n{'='*70}")
    print(f"  PROBE: {policy_name}")
    print(f"{'='*70}")

    env_cpp = flightgym.QuadrotorEnv_v1()
    env = FlightmareSB3VecEnv(env_cpp)
    obs = env.reset()

    print(f"  Num envs: {env.num_envs}, Obs dim: {obs.shape[1]}, Act dim: {env.action_space.shape[0]}")
    print(f"  Initial obs[0]: {obs[0][:6]}... (first 6 dims = lin_vel + lin_acc)")
    print()

    cumulative_rewards = np.zeros(env.num_envs)
    episode_lengths = np.zeros(env.num_envs, dtype=int)
    total_dones = 0
    step_rewards = []

    for step in range(n_steps):
        action = action_fn(obs, env)
        obs, reward, done, info = env.step(action)

        cumulative_rewards += reward
        episode_lengths += 1
        total_dones += done.sum()

        step_rewards.append(reward.copy())

        # Reset tracking for done envs
        for i in range(env.num_envs):
            if done[i]:
                cumulative_rewards[i] = 0
                episode_lengths[i] = 0

        # Log every 10 steps for first few envs
        if step % 10 == 0 and step < 100:
            r_str = ", ".join([f"{reward[i]:.4f}" for i in range(min(n_envs_to_show, env.num_envs))])
            cum_str = ", ".join([f"{cumulative_rewards[i]:.2f}" for i in range(min(n_envs_to_show, env.num_envs))])
            print(f"  Step {step:3d}: reward=[{r_str}] | cumul=[{cum_str}] | dones={done.sum()}")

        # Log gate observation dims for env 0
        if step == 0:
            print(f"\n  === Observation Breakdown (env 0, step 0) ===")
            print(f"  Lin vel (0:3):     {obs[0, 0:3]}")
            print(f"  Lin acc (3:6):     {obs[0, 3:6]}")
            print(f"  Rot mat (6:15):    {obs[0, 6:15]}")
            print(f"  Ang vel (15:18):   {obs[0, 15:18]}")
            print(f"  Gate1 (18:22):     r={obs[0, 18]:.3f}, θ={obs[0, 19]:.3f}, φ={obs[0, 20]:.3f}, α={obs[0, 21]:.3f}")
            print(f"  Gate2 (22:26):     r={obs[0, 22]:.3f}, θ={obs[0, 23]:.3f}, φ={obs[0, 24]:.3f}, α={obs[0, 25]:.3f}")
            print()

    # Summary statistics
    all_rewards = np.array(step_rewards)  # (n_steps, n_envs)
    print(f"\n  === Summary ({n_steps} steps) ===")
    print(f"  Reward per step:  mean={all_rewards.mean():.5f}, std={all_rewards.std():.5f}")
    print(f"                    min={all_rewards.min():.5f}, max={all_rewards.max():.5f}")
    print(f"  Total dones:      {total_dones} (across {env.num_envs} envs)")
    print(f"  Done rate:        {total_dones / n_steps:.2f} per step")

    # Check for pathological reward patterns
    mean_per_step = all_rewards.mean(axis=1)
    if np.all(np.abs(mean_per_step) < 1e-6):
        print(f"\n  ⚠️ WARNING: Reward is essentially zero for all steps!")
        print(f"     Progress reward may not be computed correctly.")
    elif np.mean(mean_per_step) < -0.1:
        print(f"\n  ⚠️ WARNING: Mean reward is significantly negative ({np.mean(mean_per_step):.4f})")
        print(f"     The drone is being penalized more than rewarded.")
    elif np.mean(mean_per_step) > 0:
        print(f"\n  ✅ Mean reward is positive ({np.mean(mean_per_step):.4f}) — progress is working!")

    # Check if Gate 1 distance (obs dim 18) is changing
    print(f"\n  === Gate Distance Tracking (env 0) ===")
    print(f"  Step 0:   Gate1 dist = {obs[0, 18]:.3f}")
    env.close()


def hover_policy(obs, env):
    """All-zero actions = hover thrust (after denormalization by C++)."""
    return np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)


def random_policy(obs, env):
    """Random actions in [-1, 1]."""
    return np.random.uniform(-1, 1, size=(env.num_envs, env.action_space.shape[0])).astype(np.float32)


def gentle_forward_policy(obs, env):
    """Slightly positive thrust to move forward."""
    # All motors at slight positive offset from hover
    action = np.ones((env.num_envs, env.action_space.shape[0]), dtype=np.float32) * 0.1
    return action


def main():
    print("=" * 70)
    print("  REWARD FUNCTION DIAGNOSTIC PROBE")
    print("  Tests whether reward increases as drone approaches gates")
    print("=" * 70)

    # Test 1: Hover — reward should be near zero (no progress)
    run_probe("Hover (zero action)", hover_policy, n_steps=200)

    # Test 2: Random — reward statistics
    run_probe("Random actions", random_policy, n_steps=200)

    # Test 3: Gentle forward — should show some progress
    run_probe("Gentle forward thrust", gentle_forward_policy, n_steps=200)

    print("\n\n" + "=" * 70)
    print("  PROBE COMPLETE")
    print("=" * 70)
    print("\nKey things to check:")
    print("  1. Does hover produce near-zero reward? (progress should be ~0)")
    print("  2. Does random produce mostly negative reward? (crashes + backward progress)")
    print("  3. Is the gate distance (obs dim 18) reasonable? (should be ~10-40m)")
    print("  4. How many dones per step? (>0.5 = too many crashes)")
    print("  5. Is reward magnitude reasonable? (should be -1 to +1 range per step)")


if __name__ == "__main__":
    main()
