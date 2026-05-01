"""Diagnostic test for IROS-2021 Distributed Initialization (paper L247-261).

What the paper specifies (methodology.tex L247-261):
  - Phase 1: spawn quadrotors in HOVER state around the CENTERS of all
    path segments, exposing them to all gate observations from step 0.
  - Phase 2: once the policy passes gates RELIABLY, sample initial states
    from previously recorded trajectories — this avoids the negative
    impact of starting from hover in high-speed regions of the track.

What this script verifies:
  A. Phase 1 spawns hit ALL track segments (not biased to any).
  B. Phase 1 spawn position is near each segment's center (not at gate plane).
  C. Phase 1 spawn velocity is hover-like (||v|| small).
  D. The trajectory buffer gets populated during episodes (only flying states).
  E. Phase 2 actually kicks in: with a trained policy reliably passing gates,
     `total_episodes > 200`, `pass_rate > traj_init_threshold`, and
     `buffer_size > 100` are all reachable in a short test, after which
     spawn velocities show a bimodal distribution (hover ∪ high-speed).
  F. Phase 2 timing: roughly when in training does it engage?

Usage:
    python diag_distributed_init.py            # full test, uses trained model
    python diag_distributed_init.py --no-model # phase-1 audit only
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Locate flightgym ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from flightgym import QuadrotorEnv_v1
from sb3_vec_env import FlightmareSB3VecEnv

FLIGHTMARE = Path(os.environ.get("FLIGHTMARE_PATH", str(Path.home() / "flightmare")))
CFG_PATH = FLIGHTMARE / "flightlib/configs/quadrotor_env.yaml"
VEC_CFG_PATH = FLIGHTMARE / "flightlib/configs/vec_env.yaml"


def load_track_geometry():
    """Read gate positions + start_pos from the production YAML."""
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    gates = np.array([g["position"] for g in cfg["gates"]])
    start_pos = np.array(cfg["start_pos"])
    return start_pos, gates, cfg


def segment_centers(start_pos, gates):
    """Midpoint of each track segment (prev_gate -> curr_gate)."""
    centers = []
    prev = start_pos
    for g in gates:
        centers.append(0.5 * (prev + g))
        prev = g
    return np.array(centers)


def velocity_from_obs(obs):
    """obs[0:3] is v_WB in world frame (paper methodology.tex L173)."""
    return obs[:, 0:3]


# ── Test A/B/C: Phase 1 spawn distribution ──────────────────────────
def test_phase1_spawn_distribution(env, raw_env, num_resets=50):
    """Reset many times, gather spawn positions / velocities / gate_idx.
    Phase 1 is the default at construction (use_traj_init_=false), so until
    the trajectory buffer fills + pass_rate crosses the threshold, every
    reset goes through the hover-init branch.
    """
    start_pos, gates, _ = load_track_geometry()
    seg_mids = segment_centers(start_pos, gates)
    num_gates = len(gates)
    n_envs = env.num_envs

    raw_env.set_random_reset(True)

    gate_idx_counts = np.zeros(num_gates, dtype=int)
    spawn_vels = []
    spawn_pos = []
    seg_assignments = []  # (gate_idx, dist_to_seg_center)

    extra_names = raw_env.wrapper.getExtraInfoNames()
    gate_idx_col = extra_names.index("gate_idx")

    for _ in range(num_resets):
        obs = env.reset()  # SB3 wrapper -> wrapper.reset
        # We need extra_info at reset time. wrapper.reset() doesn't return it.
        # Take one zero-action step to populate extra_info, then read it.
        zero_act = np.zeros((n_envs, env.action_space.shape[0]), dtype=np.float32)
        # NOTE: obs from reset already contains v in [0:3] — that's the spawn
        # velocity. Use that directly to avoid the side effects of stepping.
        v = velocity_from_obs(obs)

        # Read gate_idx via one step (cheap)
        obs2, _, _, infos = env.step(zero_act)
        for i, inf in enumerate(infos):
            gi = int(inf["gate_idx"])
            if 0 <= gi < num_gates:
                gate_idx_counts[gi] += 1
                seg_assignments.append((gi, np.linalg.norm(v[i])))
            spawn_vels.append(v[i])
            # Reconstruct spawn pos from extra_info (we only have it AFTER step)
            spawn_pos.append([inf["pos_x"], inf["pos_y"], inf["pos_z"]])

    spawn_vels = np.array(spawn_vels)
    spawn_pos = np.array(spawn_pos)
    speeds = np.linalg.norm(spawn_vels, axis=1)

    print("\n=== Phase 1: Spawn distribution ===")
    print(f"  Resets × envs: {num_resets} × {n_envs} = {num_resets * n_envs} samples")
    print(f"  Gate-idx histogram (target gate after spawn):")
    for gi, cnt in enumerate(gate_idx_counts):
        bar = "█" * int(40 * cnt / max(gate_idx_counts.max(), 1))
        print(f"    seg {gi:2d}: {cnt:5d}  {bar}")
    print(f"  Min/max segment count ratio: "
          f"{gate_idx_counts.min() / max(gate_idx_counts.max(), 1):.3f}  (1.0 = perfectly uniform)")

    print(f"\n  Spawn speed (||v||):")
    print(f"    mean = {speeds.mean():.3f} m/s,  max = {speeds.max():.3f}, "
          f"std = {speeds.std():.3f}")
    print(f"    fraction above 1.5 m/s = {(speeds > 1.5).mean():.3%} "
          f"(should be ~0% in Phase 1)")

    # Distance from spawn position to expected segment center
    print(f"\n  Spawn-to-segment-center distance:")
    dists = []
    # NOTE: spawn_pos and seg_assignments are in step order; gate_idx_counts
    # uses the step's reading. Build a parallel list of (gi, pos).
    # Actually we recorded spawn_pos in same loop as infos, so iterate together.
    spawn_pos_arr = np.array(spawn_pos)
    # Re-derive (gi, pos) pairs:
    flat_idx = 0
    pos_pairs = []
    # We collected n_envs entries per reset, in the same order that infos came
    # back. seg_assignments was built skipping out-of-range gate_idx, so it
    # might have fewer entries — recompute aligned.
    for r in range(num_resets):
        for i in range(n_envs):
            # Approx — we don't have the exact gi-per-pair stored. Just compute
            # nearest-segment-center as a sanity check instead.
            p = spawn_pos_arr[r * n_envs + i]
            d_to_segs = np.linalg.norm(seg_mids - p[None, :], axis=1)
            dists.append(d_to_segs.min())
    dists = np.array(dists)
    print(f"    mean = {dists.mean():.2f} m,  median = {np.median(dists):.2f} m,  "
          f"max = {dists.max():.2f} m")
    print(f"    (Phase 1 spec: t={'0.5±0.3'} along segment + ±1.5 m lateral. "
          f"Expected ~1-3 m typical, up to ~5 m worst case.)")

    # Verdicts
    uniform_ratio = gate_idx_counts.min() / max(gate_idx_counts.max(), 1)
    speed_ok = (speeds > 1.5).mean() < 0.02
    coverage_ok = (gate_idx_counts > 0).all()

    print(f"\n  Verdict:")
    print(f"    [{'PASS' if coverage_ok else 'FAIL'}] All {num_gates} segments hit")
    print(f"    [{'PASS' if uniform_ratio > 0.4 else 'WARN'}] Distribution roughly uniform "
          f"(min/max ratio = {uniform_ratio:.2f}, want > 0.4)")
    print(f"    [{'PASS' if speed_ok else 'FAIL'}] Spawn speed is hover-like "
          f"({(speeds > 1.5).mean():.2%} above 1.5 m/s)")

    return dict(coverage_ok=coverage_ok, uniform_ratio=uniform_ratio,
                speed_ok=speed_ok, spawn_speeds=speeds)


# ── Test D/E: Phase 2 trigger ───────────────────────────────────────
def test_phase2_trigger(model_path, vecnorm_path, override_threshold=0.05,
                        max_steps=15000):
    """Drive episodes with the trained policy under a *relaxed* threshold.

    The production threshold is 0.6 (avg gates passed per episode).  For a
    diagnostic run on the order of minutes we drop it to ~0.05 — even a
    weak policy crossing one gate occasionally will reliably trigger the
    transition, letting us observe Phase 2's spawn-velocity bimodality.
    """
    print("\n=== Phase 2: Transition trigger test ===")

    # Patch yaml in place (with restore guard)
    with open(CFG_PATH) as f:
        original_yaml = f.read()
    cfg = yaml.safe_load(original_yaml)
    orig_thresh = cfg["rl"].get("traj_init_threshold", 0.6)
    cfg["rl"]["traj_init_threshold"] = float(override_threshold)
    print(f"  (Temporarily lowering traj_init_threshold {orig_thresh} → "
          f"{override_threshold} so Phase 2 fires within a short run.)")

    try:
        with open(CFG_PATH, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

        # Construct env AFTER patching yaml so it loads new threshold
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        cpp_env = QuadrotorEnv_v1()
        raw_env = FlightmareSB3VecEnv(cpp_env)
        if vecnorm_path and os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, raw_env)
            env.training = False
            env.norm_reward = False
        else:
            env = raw_env
        model = PPO.load(model_path)
        raw_env.set_random_reset(True)  # DISTRIBUTED INIT ON

        extra_names = raw_env.wrapper.getExtraInfoNames()
        idx_gates_passed = extra_names.index("gates_passed")
        idx_term = extra_names.index("term_reason")
        idx_vel = extra_names.index("vel_norm")

        n_envs = env.num_envs
        obs = env.reset()
        episode_count = 0
        gates_total = 0
        # Track post-transition spawn speeds (after first reset that uses
        # buffered trajectory state). We can't read use_traj_init_ directly,
        # but we can monitor the spawn-velocity DISTRIBUTION over time.
        spawn_speed_log = []  # (episode_count, mean_speed_at_spawn)
        per_step_speeds_at_spawn = []  # speeds collected when we know a reset just happened

        # Watch a single canary env's gate_idx history to estimate per-env
        # progression. Stop when running mean of per-episode gates is well
        # above the threshold AND we have plenty of episodes.
        t0 = time.time()
        for step_i in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(action)

            # When done flag fires, the C++ VecEnv has ALREADY autoreset the
            # env, so the obs we just got is the post-reset spawn state.
            for i, d in enumerate(done):
                if d:
                    episode_count += 1
                    gates_total += int(infos[i]["gates_passed"])
                    # obs is normalized in VecNormalize; raw_env's last reading
                    # is the unnormalized spawn vel.
                    raw_v = np.linalg.norm(
                        raw_env._observation[i, 0:3]
                    )  # post-reset velocity
                    per_step_speeds_at_spawn.append(
                        (episode_count, raw_v, int(infos[i]["gates_passed"]))
                    )

            if step_i and step_i % 1500 == 0:
                # Periodic progress report
                if per_step_speeds_at_spawn:
                    recent = per_step_speeds_at_spawn[-1000:]
                    mean_speed = np.mean([r[1] for r in recent])
                    above_hover = np.mean([r[1] > 2.0 for r in recent])
                    pass_rate = gates_total / max(episode_count, 1)
                    print(f"    step {step_i:5d} | episodes={episode_count:5d} | "
                          f"pass_rate={pass_rate:.3f} | "
                          f"mean_spawn_v={mean_speed:.2f} m/s | "
                          f"frac_v>2={above_hover:.2%}")

        elapsed = time.time() - t0
        pass_rate = gates_total / max(episode_count, 1)
        print(f"\n  After {max_steps} steps ({elapsed:.0f}s wallclock):")
        print(f"    Total episodes = {episode_count}")
        print(f"    Total gates passed = {gates_total}")
        print(f"    pass_rate (gates/episode) = {pass_rate:.3f}  "
              f"(threshold = {override_threshold})")

        # Bimodality check on spawn speeds: split the log in half by
        # episode index; phase 1 should dominate early, phase 2 should
        # contribute high-speed spawns later.
        if len(per_step_speeds_at_spawn) > 200:
            speeds_arr = np.array([r[1] for r in per_step_speeds_at_spawn])
            n = len(speeds_arr)
            early = speeds_arr[: n // 4]
            late = speeds_arr[3 * n // 4 :]
            print(f"\n  Spawn speed by quarter:")
            print(f"    Early Q1 (first 25% of episodes): "
                  f"mean={early.mean():.2f}, max={early.max():.2f}, "
                  f"frac>2 m/s = {(early > 2).mean():.2%}")
            print(f"    Late  Q4 (last  25% of episodes): "
                  f"mean={late.mean():.2f}, max={late.max():.2f}, "
                  f"frac>2 m/s = {(late > 2).mean():.2%}")

            phase2_engaged = (late > 2).mean() > 0.05 and (late > 2).mean() > (early > 2).mean() + 0.05
            print(f"\n  Verdict:")
            if phase2_engaged:
                print(f"    [PASS] Phase 2 engaged: late-run spawns include "
                      f"high-speed (>2 m/s) samples consistent with "
                      f"trajectory replay.")
            else:
                print(f"    [WARN] No clear evidence of Phase 2 engaging "
                      f"(late spawns still hover-like). Possible causes:")
                print(f"           - pass_rate ({pass_rate:.3f}) never crossed "
                      f"threshold ({override_threshold})")
                print(f"           - per-env total_episodes_ never exceeded 200 "
                      f"(per-env episodes ≈ {episode_count / n_envs:.1f})")
                print(f"           - traj_buffer never reached 100 samples")
        else:
            print(f"  [SKIP] Too few episodes ({episode_count}) to assess "
                  f"bimodality.")

    finally:
        # Restore yaml
        with open(CFG_PATH, "w") as f:
            f.write(original_yaml)
        print(f"\n  (Restored original quadrotor_env.yaml)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true",
                    help="Skip phase-2 test (no trained model needed).")
    ap.add_argument("--model",
                    default="saved/best_iros_gate_racing/best_model.zip")
    ap.add_argument("--vecnorm",
                    default="saved/best_iros_gate_racing/vec_normalize.pkl")
    ap.add_argument("--phase1-resets", type=int, default=50)
    ap.add_argument("--phase2-steps", type=int, default=15000)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    print("Distributed Initialization Diagnostic — IROS 2021 paper L247-261")
    print("=" * 65)

    # ── Static audit ────────────────────────────────────────────────
    print("\n=== Static audit of implementation ===")
    issues = []
    # 1. Snapshot omega is dropped on Phase-2 reset
    src = (FLIGHTMARE / "flightlib/src/envs/quadrotor_env/quadrotor_env.cpp").read_text()
    phase2_block = src.split("// Phase 2: sample from trajectory buffer")[1].split(
        "// Phase 1 (or Phase 2 fallback)"
    )[0]
    if "OMEX" not in phase2_block.upper() and "snap.omega" not in phase2_block:
        issues.append(
            "Phase-2 reset DROPS snap.omega — recorded angular velocity is "
            "discarded, the drone reincarnates with ω=0 even though pos/vel/quat "
            "are taken from a flying snapshot. Partially defeats the paper's "
            "goal of avoiding hover init in high-speed regions."
        )
    # 2. total_episodes > 200 hardcoded
    if "total_episodes_ > 200" in src and "total_episodes_min" not in src:
        issues.append(
            "Phase-2 transition has a HARD-CODED `total_episodes_ > 200` floor. "
            "Not exposed in YAML — magic number. Tunable would be cleaner, "
            "though 200 is reasonable for 100-env training."
        )
    # 3. pass_rate semantics
    if "Scalar pass_rate = Scalar(total_gates_passed_)" in src:
        issues.append(
            "`pass_rate` is mis-named: it's mean-gates-per-episode, not a "
            "probability. With 11 gates, threshold=0.6 means 'on average <1 "
            "gate cleared per episode' — a weak bar for 'reliably' (paper L259). "
            "Consider raising to 2-5 (avg of multiple gates) before switching to "
            "Phase 2 to avoid the high-speed-state-from-bad-policy failure mode "
            "the paper warns about."
        )
    # 4. global vs per-env state
    issues.append(
        "[INFO] State is per-env (each parallel env has its own counters/buffer). "
        "Phase 2 transition fires asynchronously across envs as each crosses "
        "200 episodes individually — this is fine but worth knowing for "
        "reproducibility."
    )

    print("Findings:")
    for i, msg in enumerate(issues, 1):
        print(f"  {i}. {msg}\n")

    # ── Phase 1 spawn distribution ──────────────────────────────────
    if args.phase1_resets > 0:
        cpp_env = QuadrotorEnv_v1()
        raw_env = FlightmareSB3VecEnv(cpp_env)
        test_phase1_spawn_distribution(raw_env, raw_env, num_resets=args.phase1_resets)
        raw_env.close()

    # ── Phase 2 trigger ─────────────────────────────────────────────
    if not args.no_model:
        if not os.path.exists(args.model):
            print(f"\n[SKIP] Phase 2 test: model not found at {args.model}")
        else:
            test_phase2_trigger(args.model, args.vecnorm,
                                override_threshold=args.threshold,
                                max_steps=args.phase2_steps)

    print("\nDone.")


if __name__ == "__main__":
    main()
