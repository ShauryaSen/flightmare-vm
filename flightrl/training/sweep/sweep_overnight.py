#!/usr/bin/env python3
"""Overnight hyperparameter sweep orchestrator.

For each condition:
  1. Patch flightlib/configs/quadrotor_env.yaml with the condition's `rl` overrides
  2. Spawn a subprocess running _train_one.py for `--total-timesteps` steps
  3. Subprocess writes result.json + progress.json + best_model/ + tb/
  4. Restore baseline YAML before next condition (always, even on failure)

Each subprocess gets a fresh Python interpreter so the C++ env reloads its
YAML cleanly. Crashes in one condition do not take down the sweep.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ruamel.yaml import YAML

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
FLIGHTMARE_ROOT = TRAINING_DIR.parent.parent
ENV_YAML = FLIGHTMARE_ROOT / "flightlib" / "configs" / "quadrotor_env.yaml"
SWEEP_RUNS_DIR = THIS_DIR / "runs"
BACKUP_YAML = THIS_DIR / "baseline_yaml_backup.yaml"

sys.path.insert(0, str(THIS_DIR))
from conditions import CONDITIONS  # noqa: E402

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=2, offset=0)


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.load(f)


def dump_yaml(data, path: Path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def patch_yaml(baseline_data, rl_overrides: dict):
    """Return a new YAML dict with rl_overrides applied to the rl section."""
    import copy
    data = copy.deepcopy(baseline_data)
    if rl_overrides:
        if "rl" not in data:
            data["rl"] = {}
        for k, v in rl_overrides.items():
            data["rl"][k] = v
    return data


def preflight_checks(args):
    print("=" * 70, flush=True)
    print("  PRE-FLIGHT CHECKS", flush=True)
    print("=" * 70, flush=True)

    # 1. YAML exists
    if not ENV_YAML.exists():
        sys.exit(f"FATAL: {ENV_YAML} not found")
    print(f"  [ok] config:  {ENV_YAML}", flush=True)

    # 2. Free disk
    free_gb = shutil.disk_usage(THIS_DIR).free / (1024**3)
    print(f"  [.. ] disk:    {free_gb:.1f} GB free", flush=True)
    if free_gb < 8:
        print(f"  [WARN] less than 8 GB free — sweep may fill disk", flush=True)

    # 3. Cores + RAM
    try:
        n_cores = os.cpu_count()
        print(f"  [.. ] cores:   {n_cores}", flush=True)
        if n_cores < 6:
            print(f"  [WARN] only {n_cores} cores — bump UTM allocation to 8 if possible",
                  flush=True)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    print(f"  [.. ] memory:  {kb / (1024**2):.1f} GiB", flush=True)
                    break
    except Exception:
        pass

    # 4. flightgym imports
    try:
        import flightgym  # noqa: F401
        print(f"  [ok] flightgym imports", flush=True)
    except Exception as e:
        sys.exit(f"FATAL: flightgym import failed: {e}")

    # 5. Backup exists or create
    if BACKUP_YAML.exists() and not args.force_backup:
        print(f"  [ok] backup:  {BACKUP_YAML} (existing)", flush=True)
    else:
        shutil.copy(ENV_YAML, BACKUP_YAML)
        print(f"  [ok] backup:  {BACKUP_YAML} (created)", flush=True)

    # 6. Output dir
    SWEEP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [ok] output:  {SWEEP_RUNS_DIR}", flush=True)

    print("=" * 70, flush=True)


def run_one_condition(cond, total_timesteps, log_every, seed):
    """Patch YAML, spawn subprocess, return (success, result_dict_or_none)."""
    name = cond["name"]
    run_dir = SWEEP_RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patch YAML
    baseline = load_yaml(BACKUP_YAML)
    patched = patch_yaml(baseline, cond.get("rl", {}))
    dump_yaml(patched, ENV_YAML)

    # Save a copy of what we used inside the run dir for traceability
    dump_yaml(patched, run_dir / "quadrotor_env.yaml.used")
    with open(run_dir / "condition.json", "w") as f:
        json.dump(cond, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  rl overrides:  {cond.get('rl', {})}", flush=True)
    print(f"  ppo overrides: {cond.get('ppo', {})}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # 2. Build subprocess command
    cmd = [
        sys.executable, "-u", str(THIS_DIR / "_train_one.py"),
        str(run_dir),
        "--total-timesteps", str(total_timesteps),
        "--log-every", str(log_every),
        "--seed", str(seed),
    ]
    ppo_over = cond.get("ppo", {})
    if "ent_coef" in ppo_over:
        cmd += ["--ent-coef", str(ppo_over["ent_coef"])]
    if "learning_rate" in ppo_over:
        cmd += ["--learning-rate", str(ppo_over["learning_rate"])]

    # 3. Run subprocess. Stdout streams to our stdout (which goes to log file
    # via launch.sh nohup). Per-run log also captured.
    log_path = run_dir / "train.log"
    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=1,
                                    universal_newlines=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            rc = proc.wait()
    except Exception as e:
        print(f"  [ERR] subprocess launch failed: {e}", flush=True)
        return False, None

    elapsed = time.time() - t0
    if rc != 0:
        print(f"  [FAIL] {name} exited with code {rc} after {elapsed/60:.1f} min",
              flush=True)
        # Still try to read partial result
    else:
        print(f"  [done] {name} finished in {elapsed/60:.1f} min", flush=True)

    # 4. Read result.json if present
    result_path = run_dir / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            return rc == 0, json.load(f)
    return False, None


def restore_baseline_yaml():
    if BACKUP_YAML.exists():
        shutil.copy(BACKUP_YAML, ENV_YAML)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=25_000_000,
                        help="Steps per condition")
    parser.add_argument("--log-every", type=int, default=100_000,
                        help="Progress.json snapshot frequency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated condition names to run (default: all)")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated condition names to skip")
    parser.add_argument("--force-backup", action="store_true",
                        help="Overwrite existing baseline_yaml_backup.yaml")
    args = parser.parse_args()

    preflight_checks(args)

    # Filter conditions
    conds = list(CONDITIONS)
    if args.only:
        names = set(args.only.split(","))
        conds = [c for c in conds if c["name"] in names]
        print(f"\nFiltered to {len(conds)} conditions: {[c['name'] for c in conds]}",
              flush=True)
    if args.skip:
        names = set(args.skip.split(","))
        conds = [c for c in conds if c["name"] not in names]

    print(f"\nSweep plan:", flush=True)
    print(f"  conditions:       {len(conds)}", flush=True)
    print(f"  steps/condition:  {args.total_timesteps:,}", flush=True)
    print(f"  total steps:      {args.total_timesteps * len(conds):,}", flush=True)
    print(f"  est wall-time:    {args.total_timesteps * len(conds) / 18_000 / 60:.0f} min "
          f"(at 18K FPS) to {args.total_timesteps * len(conds) / 12_000 / 60:.0f} min "
          f"(at 12K FPS)", flush=True)

    sweep_t0 = time.time()
    summary = []
    try:
        for i, cond in enumerate(conds):
            print(f"\n[{i+1}/{len(conds)}] {cond['name']}", flush=True)
            success, result = run_one_condition(
                cond, args.total_timesteps, args.log_every, args.seed)
            summary.append({
                "name": cond["name"],
                "success": success,
                "result": result,
            })
            # Always restore YAML between runs in case the next iteration fails
            # before patching
            restore_baseline_yaml()

            # Save running summary so partial overnight progress is recoverable
            with open(THIS_DIR / "sweep_summary.json", "w") as f:
                json.dump({
                    "started_at": sweep_t0,
                    "elapsed_min": (time.time() - sweep_t0) / 60,
                    "completed": i + 1,
                    "total": len(conds),
                    "runs": summary,
                }, f, indent=2)
    finally:
        # Belt-and-suspenders: restore baseline no matter what
        restore_baseline_yaml()
        print(f"\n[final] baseline YAML restored from {BACKUP_YAML}", flush=True)

    elapsed_min = (time.time() - sweep_t0) / 60
    print(f"\n{'=' * 70}", flush=True)
    print(f"  SWEEP COMPLETE  —  {elapsed_min:.1f} min total", flush=True)
    print(f"  Run `python sweep/sweep_report.py` to see ranked results.", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
