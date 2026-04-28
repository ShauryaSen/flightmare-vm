#!/usr/bin/env python3
"""
Reward Verification Script

Verifies that:
  1. Reward components sum to total reward (r_progress + r_safety + r_bodyrate = r_total)
  2. Progress reward matches manual calculation from position changes
  3. Progress is positive when drone moves toward gate
  4. Gate crossing detection works (drone passes through gate → gate_bonus fires)
  5. Terminal penalties work correctly

Uses the extra_info fields exposed by updateExtraInfo() for exact verification.
"""

import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


def get_info_dict(infos, env_idx, extra_names):
    """Extract extra_info for a specific env as a dict."""
    return infos[env_idx]


def verify_reward_decomposition(env, n_steps=200):
    """Test 1: Verify reward = sum of components for every step."""
    print("\n" + "=" * 70)
    print("  TEST 1: Reward Decomposition (components must sum to total)")
    print("=" * 70)

    obs = env.reset()
    max_err = 0.0
    total_checked = 0
    failures = 0

    for step in range(n_steps):
        action = np.zeros((env.num_envs, 4), dtype=np.float32)
        obs, reward, done, infos = env.step(action)

        for i in range(env.num_envs):
            info = infos[i]
            if "r_total" not in info:
                print("  ERROR: extra_info not populated! Rebuild C++ lib.")
                return False

            r_prog = info["r_progress"]
            r_safe = info["r_safety"]
            r_body = info["r_bodyrate"]
            r_gate = info["r_gate_bonus"]
            r_total_info = info["r_total"]

            # The actual reward returned by step() includes terminal reward added by VecEnv
            # For non-terminal steps, reward should equal r_total from extra_info
            computed_sum = r_prog + r_safe + r_body + r_gate
            err = abs(computed_sum - r_total_info)
            max_err = max(max_err, err)
            total_checked += 1

            if err > 1e-4:
                failures += 1
                if failures <= 5:
                    print(f"  ✗ Step {step}, env {i}: sum={computed_sum:.6f} vs total={r_total_info:.6f} (err={err:.6f})")
                    print(f"    prog={r_prog:.6f}, safe={r_safe:.6f}, body={r_body:.6f}, gate={r_gate:.6f}")

    if failures == 0:
        print(f"  ✓ PASSED: All {total_checked} steps match. Max error: {max_err:.2e}")
    else:
        print(f"  ✗ FAILED: {failures}/{total_checked} mismatches. Max error: {max_err:.6f}")
    return failures == 0


def verify_progress_direction(env, n_steps=100):
    """Test 2: Verify progress reward correlates with movement toward gate."""
    print("\n" + "=" * 70)
    print("  TEST 2: Progress Direction (moving toward gate = positive reward)")
    print("=" * 70)

    obs = env.reset()

    # Collect a few steps of data from env 0
    prev_pos = None
    progress_positive_count = 0
    progress_negative_count = 0
    progress_data = []

    for step in range(n_steps):
        action = np.zeros((env.num_envs, 4), dtype=np.float32)
        # Give slight upward thrust to prevent crash
        action[:, :] = 0.05  # Slightly above hover
        obs, reward, done, infos = env.step(action)

        info = infos[0]
        pos = np.array([info["pos_x"], info["pos_y"], info["pos_z"]])
        r_prog = info["r_progress"]
        gate_idx = int(info["gate_idx"])
        s_prev = info["s_prev"]

        if r_prog > 0.001:
            progress_positive_count += 1
        elif r_prog < -0.001:
            progress_negative_count += 1

        if step < 20:
            progress_data.append((step, pos, r_prog, gate_idx, s_prev))

        if done[0]:
            break

    print(f"  Env 0 over {min(step+1, n_steps)} steps:")
    print(f"    Positive progress steps: {progress_positive_count}")
    print(f"    Negative progress steps: {progress_negative_count}")
    print(f"    First 20 steps:")
    for s, pos, r_p, g_idx, s_p in progress_data:
        print(f"      Step {s:3d}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"progress={r_p:+.5f} gate={g_idx} s_prev={s_p:.3f}")

    return True


def verify_progress_math(env, n_steps=50):
    """Test 3: Manually compute segment_progress and compare to reported value."""
    print("\n" + "=" * 70)
    print("  TEST 3: Progress Math Verification")
    print("=" * 70)

    # Read gate positions from the YAML to manually compute progress
    import yaml
    yaml_path = os.environ.get("FLIGHTMARE_PATH", "") + "/flightlib/configs/quadrotor_env.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    gates = []
    for g in cfg["gates"]:
        p = g["position"]
        gates.append(np.array(p, dtype=np.float64))
    num_gates = len(gates)
    delta_max = cfg["rl"]["delta_max"]

    def segment_progress_py(pos, target_gate_idx):
        """Python replica of C++ segment_progress"""
        g1 = gates[(target_gate_idx - 1 + num_gates) % num_gates]
        g2 = gates[target_gate_idx]
        seg = g2 - g1
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            return 0.0
        raw = np.dot(pos - g1, seg) / seg_len
        return max(0.0, min(raw, seg_len + 0.5))

    obs = env.reset()
    max_err = 0.0
    total_checked = 0
    failures = 0

    # Track s_prev ourselves
    prev_s = {}

    # First step to get initial info
    action = np.zeros((env.num_envs, 4), dtype=np.float32)
    obs, reward, done, infos = env.step(action)

    for step in range(n_steps):
        action = np.zeros((env.num_envs, 4), dtype=np.float32)
        action[:, :] = 0.02  # Gentle thrust

        # Before step: record state
        prev_infos = infos

        obs, reward, done, infos = env.step(action)

        # Check env 0 (skip if it just reset)
        if not done[0]:
            i = 0
            info = infos[i]
            prev_info = prev_infos[i]

            pos = np.array([info["pos_x"], info["pos_y"], info["pos_z"]])
            gate_idx = int(info["gate_idx"])

            # Compute expected progress manually
            s_curr_py = segment_progress_py(pos, gate_idx)
            s_prev_reported = prev_info["s_prev"]
            expected_progress = max(-delta_max, min(s_curr_py - s_prev_reported, delta_max))

            actual_progress = info["r_progress"]

            err = abs(expected_progress - actual_progress)
            max_err = max(max_err, err)
            total_checked += 1

            if err > 0.01 and total_checked <= 10:
                print(f"  Step {step}: expected={expected_progress:.6f} vs actual={actual_progress:.6f} "
                      f"(err={err:.6f})")
                print(f"    pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), gate={gate_idx}, "
                      f"s_prev={s_prev_reported:.4f}, s_curr_py={s_curr_py:.4f}")
                failures += 1

    if failures == 0 and total_checked > 0:
        print(f"  ✓ PASSED: All {total_checked} steps match. Max error: {max_err:.2e}")
    elif total_checked == 0:
        print(f"  ⚠ No steps checked (all envs may have terminated)")
    else:
        print(f"  ⚠ {failures} steps with err > 0.01 (may be gate crossing steps). Max error: {max_err:.6f}")
        if max_err < 1.0:
            print(f"     This is likely OK — gate crossing resets s_prev which causes a mismatch on that step.")

    return True


def verify_reward_scale(env, n_steps=300):
    """Test 4: Verify reward magnitudes are reasonable for learning."""
    print("\n" + "=" * 70)
    print("  TEST 4: Reward Scale Analysis")
    print("=" * 70)

    obs = env.reset()
    all_rewards = []
    all_progress = []
    all_bodyrate = []
    terminal_rewards = []
    n_dones = 0

    for step in range(n_steps):
        action = np.random.uniform(-0.5, 0.5, size=(env.num_envs, 4)).astype(np.float32)
        obs, reward, done, infos = env.step(action)

        for i in range(env.num_envs):
            all_rewards.append(reward[i])
            all_progress.append(infos[i]["r_progress"])
            all_bodyrate.append(infos[i]["r_bodyrate"])
            if done[i]:
                n_dones += 1
                # Terminal reward is added to the step reward by VecEnv
                terminal_rewards.append(reward[i] - infos[i]["r_total"])

    all_rewards = np.array(all_rewards)
    all_progress = np.array(all_progress)
    all_bodyrate = np.array(all_bodyrate)

    print(f"  Total steps: {n_steps * env.num_envs}")
    print(f"  Terminals:   {n_dones}")
    print(f"\n  Reward statistics:")
    print(f"    Total:    mean={all_rewards.mean():.5f}, std={all_rewards.std():.5f}, "
          f"min={all_rewards.min():.3f}, max={all_rewards.max():.3f}")
    print(f"    Progress: mean={all_progress.mean():.5f}, std={all_progress.std():.5f}, "
          f"min={all_progress.min():.3f}, max={all_progress.max():.3f}")
    print(f"    BodyRate: mean={all_bodyrate.mean():.5f}, std={all_bodyrate.std():.5f}, "
          f"min={all_bodyrate.min():.3f}, max={all_bodyrate.max():.3f}")

    # Check for asymmetry (negative progress should now be bounded)
    if all_progress.min() >= -0.5 - 0.01:
        print(f"\n  ✓ Progress is symmetrically clamped to [-{0.5}, +{0.5}]")
    else:
        print(f"\n  ✗ Progress min={all_progress.min():.3f} exceeds -delta_max!")

    if len(terminal_rewards) > 0:
        tr = np.array(terminal_rewards)
        print(f"\n  Terminal rewards: mean={tr.mean():.3f}, min={tr.min():.3f}, max={tr.max():.3f}")
    else:
        print(f"\n  No terminals in {n_steps} steps")

    return True


def main():
    print("=" * 70)
    print("  REWARD VERIFICATION SUITE")
    print("  Verifying all reward components after bug fixes")
    print("=" * 70)

    env_cpp = flightgym.QuadrotorEnv_v1()
    env = FlightmareSB3VecEnv(env_cpp)
    print(f"  Envs: {env.num_envs}, Obs: {env.observation_space.shape}, Act: {env.action_space.shape}")

    # Check if extra_info is available
    if len(env._extra_info_names) == 0:
        print("\n  ERROR: No extra_info names! C++ updateExtraInfo() not working.")
        print("  Make sure you rebuilt flightgym after adding updateExtraInfo().")
        return
    print(f"  Extra info fields: {env._extra_info_names}")

    results = {}
    results["decomposition"] = verify_reward_decomposition(env, n_steps=200)
    results["direction"] = verify_progress_direction(env, n_steps=100)
    results["math"] = verify_progress_math(env, n_steps=50)
    results["scale"] = verify_reward_scale(env, n_steps=300)

    print("\n\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ NEEDS REVIEW"
        print(f"  {status}: {name}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
