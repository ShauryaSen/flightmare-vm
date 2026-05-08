#!/usr/bin/env python3
"""Eval grid orchestrator — runs each (checkpoint, track[, perturbation])
through _eval_one.py as a SEQUENTIAL subprocess (track_loader.py mutates
quadrotor_env.yaml inter-process; concurrent eval would race).

Aggregates per-episode gate_idx into bootstrap CIs (§5 statistical analysis).

Usage:
  python run_eval_grid.py --runs-dir runs/ --out-dir runs/eval_grid/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
TRACK_LOADER = TRAINING_DIR / "track_loader.py"

# Eval-grid layout per §7. circle_8 is target; held-out set excludes the
# rungs Condition B trains on.
TARGET_TRACK = "circle_8"
HELDOUT_TRACKS = ["oval_8", "helix_8", "vertical_3", "figure_8"]
ROBUST_SIGMA = 0.5
EVAL_SEED = 12345  # fixed per §5 so per-episode bootstrap is valid across conditions


def discover_checkpoints(runs_dir: Path):
    """Return list of {label, model, vecnorm} dicts.

    Conditions A and C: runs/{A,C}_seed{S}/{final_model.zip,final_vec_normalize.pkl}
    Condition B: runs/B_seed{S}/F_circle_8/{final_model.zip,final_vec_normalize.pkl}
    """
    checkpoints = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.startswith("A_seed") or name.startswith("C_seed"):
            mdl = sub / "final_model.zip"
            vec = sub / "final_vec_normalize.pkl"
            if mdl.exists() and vec.exists():
                checkpoints.append({"label": name, "model": str(mdl),
                                    "vecnorm": str(vec)})
        elif name.startswith("B_seed"):
            f_dir = sub / "F_circle_8"
            mdl = f_dir / "final_model.zip"
            vec = f_dir / "final_vec_normalize.pkl"
            if mdl.exists() and vec.exists():
                checkpoints.append({"label": name, "model": str(mdl),
                                    "vecnorm": str(vec)})
    return checkpoints


def apply_track(track: str):
    cmd = [sys.executable, str(TRACK_LOADER), "apply", track, "--force"]
    r = subprocess.run(cmd, cwd=str(TRAINING_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"track_loader failed for {track}")


def run_one(model: str, vecnorm: str, track: str, perturb_sigma: float,
            out_path: Path, n_episodes: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(THIS_DIR / "_eval_one.py"),
        "--model", model,
        "--vecnorm", vecnorm,
        "--track", track,
        "--perturb-sigma", str(perturb_sigma),
        "--eval-seed", str(EVAL_SEED),
        "--n-episodes", str(n_episodes),
        "--out", str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(TRAINING_DIR))


def bootstrap_ci(arr, n_resamples: int = 10000, q=(2.5, 97.5), seed=12345):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    n = a.size
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[k] = a[idx].mean()
    lo, hi = np.percentile(means, q)
    return float(a.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True,
                   help="Root containing A_seed*/, B_seed*/, C_seed*/")
    p.add_argument("--out-dir", default=None,
                   help="Where to dump per-pair JSON + summary. "
                        "Defaults to <runs_dir>/eval_grid/")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--target", default=TARGET_TRACK)
    p.add_argument("--heldout", nargs="*", default=HELDOUT_TRACKS)
    p.add_argument("--robust-sigma", type=float, default=ROBUST_SIGMA)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir or (runs_dir / "eval_grid")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(runs_dir)
    if not checkpoints:
        print(f"No checkpoints under {runs_dir}", file=sys.stderr)
        sys.exit(2)

    # Build the (track, sigma) work list.
    pairs = [(args.target, 0.0)]
    pairs += [(t, 0.0) for t in args.heldout]
    pairs += [(t, args.robust_sigma) for t in args.heldout]

    summary = {"checkpoints": [c["label"] for c in checkpoints],
               "tracks": pairs,
               "results": {}}

    # Order outer loop by track (one apply_track call per track) for speed.
    by_track = {}
    for track, sigma in pairs:
        by_track.setdefault(track, []).append(sigma)

    for track, sigmas in by_track.items():
        print(f"=== applying track {track} ===")
        apply_track(track)
        for ck in checkpoints:
            for sigma in sigmas:
                tag = f"sigma{sigma:.2f}"
                pair_dir = out_dir / ck["label"] / track
                pair_dir.mkdir(parents=True, exist_ok=True)
                out_file = pair_dir / f"{tag}.json"
                if out_file.exists():
                    print(f"  SKIP {ck['label']} {track} {tag} (exists)")
                    continue
                print(f"  eval {ck['label']} on {track} {tag}")
                run_one(ck["model"], ck["vecnorm"], track, sigma,
                        out_file, args.n_episodes)

    # Aggregate per-condition (A/B/C) per-pair into bootstrap CIs.
    by_cond = {"A": [], "B": [], "C": []}
    for ck in checkpoints:
        cond = ck["label"][0]
        if cond not in by_cond:
            continue
        by_cond[cond].append(ck)

    rollup = {}
    for track, sigmas in by_track.items():
        for sigma in sigmas:
            key = f"{track}_sigma{sigma:.2f}"
            rollup[key] = {}
            for cond, cks in by_cond.items():
                ep_gates = []
                for ck in cks:
                    f = out_dir / ck["label"] / track / f"sigma{sigma:.2f}.json"
                    if not f.exists():
                        continue
                    d = json.loads(f.read_text())
                    ep_gates.extend(d["ep_max_gates"])
                if not ep_gates:
                    continue
                m, lo, hi = bootstrap_ci(ep_gates)
                rollup[key][cond] = {
                    "n_seeds": len(cks),
                    "n_episodes": len(ep_gates),
                    "mean_gates": m,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }

    summary["rollup"] = rollup
    with open(out_dir / "eval_grid_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== eval grid summary written to {out_dir}/eval_grid_results.json ===")
    for key, conds in rollup.items():
        print(f"\n{key}")
        for cond in ["A", "B", "C"]:
            if cond in conds:
                d = conds[cond]
                print(f"  {cond}: {d['mean_gates']:.2f} "
                      f"[{d['ci95_lo']:.2f}, {d['ci95_hi']:.2f}]  "
                      f"(n_eps={d['n_episodes']})")


if __name__ == "__main__":
    main()
