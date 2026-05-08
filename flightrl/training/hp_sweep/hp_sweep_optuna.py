#!/usr/bin/env python3
"""Optuna PPO hyperparameter sweep orchestrator.

For each trial:
  1. Sample hyperparameters from the search space (TPE sampler).
  2. Write hparams.json into a per-trial run dir.
  3. Spawn _train_one_hp.py as a subprocess.
  4. Read result.json on completion; report the composite score back to Optuna.

Composite objective: 1000 * eval_max_gate_idx + eval_mean_reward (gate count
dominates, reward breaks ties). MedianPruner stops unpromising trials early.

Stops when --max-trials is reached or --max-hours wall clock elapses.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

THIS_DIR = Path(__file__).resolve().parent
SWEEP_RUNS_DIR = THIS_DIR / "runs"
TRAIN_SCRIPT = THIS_DIR / "_train_one_hp.py"


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample a hyperparameter configuration. Search space matches param_sweep.md."""
    learning_rate = trial.suggest_categorical(
        "learning_rate", [3e-5, 1e-4, 3e-4, 1e-3])
    # n_steps=1500 and n_epochs=20 dropped — they OOM-kill on the 1.9GB-RAM
    # VM (gradient memory blows up at full-batch 300K samples × 20 epochs).
    # Trial 0 of the original sweep died at rc=-9 after 30 min in this regime.
    n_steps = trial.suggest_categorical("n_steps", [500, 750, 1000])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_categorical(
        "ent_coef", [0.0, 0.001, 0.005, 0.01, 0.02, 0.05])
    vf_coef = trial.suggest_categorical("vf_coef", [0.5, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0])
    net_arch_choice = trial.suggest_categorical(
        "net_arch", ["128_128", "256_256", "256_128", "128_128_128"])
    activation_fn = trial.suggest_categorical("activation_fn", ["Tanh", "ReLU"])

    arch_map = {
        "128_128": [128, 128],
        "256_256": [256, 256],
        "256_128": [256, 128],
        "128_128_128": [128, 128, 128],
    }

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "net_arch": arch_map[net_arch_choice],
        "activation_fn": activation_fn,
    }


def composite_score(result: dict) -> float:
    """Higher is better. Gate count dominates; reward breaks ties.

    Penalty for missing data: -1.0 sentinel (worse than any feasible score).
    """
    gate = result.get("eval_max_gate_idx", 0) or 0
    rew = result.get("eval_mean_reward", 0) or 0
    if rew != rew:  # NaN
        rew = -50.0
    return 1000.0 * float(gate) + float(rew)


class TrialRunner:
    def __init__(self, args):
        self.args = args
        self.t_start = time.time()
        SWEEP_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def __call__(self, trial: optuna.Trial) -> float:
        elapsed_h = (time.time() - self.t_start) / 3600.0
        if elapsed_h >= self.args.max_hours:
            print(f"[orchestrator] hit max-hours ({elapsed_h:.2f}h); stopping.",
                  flush=True)
            trial.study.stop()
            raise optuna.TrialPruned()

        hp = sample_hyperparameters(trial)
        trial_id = trial.number
        run_dir = SWEEP_RUNS_DIR / f"trial_{trial_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hparams_path = run_dir / "hparams.json"
        with open(hparams_path, "w") as f:
            json.dump(hp, f, indent=2)

        cmd = [
            sys.executable, "-u", str(TRAIN_SCRIPT),
            str(run_dir), str(hparams_path),
            "--total-timesteps", str(self.args.total_timesteps),
            "--seed", str(self.args.seed),
        ]
        if self.args.load_model:
            cmd += ["--load-model", self.args.load_model]
        if self.args.load_vecnorm:
            cmd += ["--load-vecnorm", self.args.load_vecnorm]

        log_path = run_dir / "trial.log"
        print(f"[orchestrator] trial {trial_id} dir={run_dir.name} hp={hp}",
              flush=True)
        print(f"  cmd: {' '.join(cmd)}", flush=True)
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

        result_path = run_dir / "result.json"
        if proc.returncode != 0 or not result_path.exists():
            print(f"[orchestrator] trial {trial_id} failed (rc={proc.returncode});"
                  f" returning sentinel score", flush=True)
            return -1.0e6

        with open(result_path) as f:
            result = json.load(f)

        score = composite_score(result)
        print(f"[orchestrator] trial {trial_id} done  score={score:.2f}  "
              f"gate={result.get('eval_max_gate_idx', 0)}  "
              f"rew={result.get('eval_mean_reward', float('nan')):.2f}",
              flush=True)
        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="ppo_hp_sweep")
    parser.add_argument("--storage", type=str,
                        default="sqlite:///hp_sweep/optuna_study.db",
                        help="Optuna SQLite path (relative to flightrl/training/)")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Optional warm-start model for every trial")
    parser.add_argument("--load-vecnorm", type=str, default=None)
    args = parser.parse_args()

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    runner = TrialRunner(args)
    print(f"[orchestrator] study={args.study_name} storage={args.storage}",
          flush=True)
    print(f"[orchestrator] max_trials={args.max_trials} max_hours={args.max_hours} "
          f"steps_per_trial={args.total_timesteps:,}", flush=True)

    try:
        study.optimize(runner, n_trials=args.max_trials,
                       gc_after_trial=True, catch=(Exception,))
    finally:
        print(f"\n{'='*60}\n  Sweep summary\n{'='*60}", flush=True)
        try:
            best = study.best_trial
            print(f"  Best trial: #{best.number}  score={best.value:.2f}", flush=True)
            print(f"  Best hparams:", flush=True)
            for k, v in best.params.items():
                print(f"    {k} = {v}", flush=True)
            best_path = THIS_DIR / "best_hparams.json"
            with open(best_path, "w") as f:
                json.dump({"trial": best.number, "score": best.value,
                           "params": best.params}, f, indent=2)
            print(f"  Saved best hparams to: {best_path}", flush=True)
        except ValueError:
            print("  No completed trials.", flush=True)


if __name__ == "__main__":
    main()
