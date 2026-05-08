#!/usr/bin/env python3
"""Round-trip test for the new applyPerturbedGates / getGates pybind methods.

Validates:
  1. getGates returns the YAML-loaded baseline.
  2. applyPerturbedGates -> getGates returns the same positions we put in.
  3. Per-env independence: mutating env 0 doesn't change env 1.
  4. The training_env.venv.wrapper attribute path used by RandomTrackCallback
     is reachable through SB3's VecNormalize (validates risk R5).

Run on the local UTM box AFTER rebuilding flightlib. Should print "ALL OK".
"""

import os
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parents[2]
EXPERIMENT_DIR = THIS.parent.parent
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

import flightgym  # noqa: E402
from sb3_vec_env import FlightmareSB3VecEnv  # noqa: E402
from track_generator import TrackGenerator  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402


def main():
    raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
    n = raw.num_envs
    n_gates = raw.wrapper.getNumGates(0)
    print(f"[smoke_pybind] num_envs={n}  num_gates={n_gates}")
    assert n_gates > 0, "C++ env reports 0 gates — check quadrotor_env.yaml"

    # 1) Baseline read.
    g0 = np.zeros((n_gates, 3), dtype=np.float32)
    raw.wrapper.getGates(0, g0)
    print(f"[smoke_pybind] baseline gates[0..2]:\n{g0[:2]}")

    # 2) Round-trip.
    target = (g0 + np.array([0.5, -0.3, 0.1], dtype=np.float32)).astype(np.float32)
    yaws = np.zeros(n_gates, dtype=np.float32)
    raw.wrapper.applyPerturbedGates(0, target, yaws)

    g1 = np.zeros((n_gates, 3), dtype=np.float32)
    raw.wrapper.getGates(0, g1)
    assert np.allclose(g1, target, atol=1e-5), \
        f"round-trip mismatch:\n  set:\n{target}\n  got:\n{g1}"
    print("[smoke_pybind] OK round-trip on env 0")

    # 3) Per-env independence — env 1 should still hold the baseline.
    g_env1 = np.zeros((n_gates, 3), dtype=np.float32)
    raw.wrapper.getGates(1, g_env1)
    assert np.allclose(g_env1, g0, atol=1e-5), \
        f"env 1 leaked from env 0 mutation:\n{g_env1}\n vs\n{g0}"
    print("[smoke_pybind] OK env 1 independent")

    # 4) Generator integration.
    yaml_path = (Path(os.environ.get("FLIGHTMARE_PATH",
                                     str(TRAINING_DIR.parents[1])))
                 / "flightlib" / "configs" / "tracks" / "linear_8.yaml")
    gen = TrackGenerator.from_baseline_yaml(yaml_path, seed=0)
    pos, yaw = gen.sample(0.5)
    raw.wrapper.applyPerturbedGates(2, pos.astype(np.float32),
                                     yaw.astype(np.float32))
    g_env2 = np.zeros((n_gates, 3), dtype=np.float32)
    raw.wrapper.getGates(2, g_env2)
    assert np.allclose(g_env2, pos, atol=1e-5), "generator -> env 2 mismatch"
    print(f"[smoke_pybind] OK generator integration (c=0.5)")

    # 5) VecNormalize attribute path (R5 mitigation).
    venv = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # SB3 nests as venv.venv -> raw_env (FlightmareSB3VecEnv)
    inner = getattr(venv, "venv", None)
    assert inner is raw, "VecNormalize.venv path differs — RandomTrackCallback must adapt"
    inner_wrapper = getattr(inner, "wrapper", None)
    assert inner_wrapper is raw.wrapper, "wrapper attribute not directly reachable"
    print("[smoke_pybind] OK VecNormalize.venv.wrapper path is valid")

    print("\nALL OK — smoke_pybind passed.")


if __name__ == "__main__":
    main()
