#!/usr/bin/env python3
"""Morning analysis script for the Optuna PPO HP sweep.

Reads the Optuna SQLite study + per-trial result.json files, prints a
ranked leaderboard, and (if matplotlib + Optuna's plotting deps are
available) emits param-importance and slice plots into hp_sweep/plots/.

Usage:
    python hp_sweep/hp_sweep_report.py
    python hp_sweep/hp_sweep_report.py --top-k 5
"""

import argparse
import json
from pathlib import Path

import optuna

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
SWEEP_RUNS_DIR = THIS_DIR / "runs"


def load_results():
    """Return a list of result dicts, one per completed trial."""
    out = []
    for d in sorted(SWEEP_RUNS_DIR.glob("trial_*")):
        rp = d / "result.json"
        if not rp.exists():
            continue
        try:
            with open(rp) as f:
                r = json.load(f)
            r["_dir"] = d.name
            out.append(r)
        except Exception:
            pass
    return out


def composite(r):
    g = r.get("eval_max_gate_idx", 0) or 0
    rew = r.get("eval_mean_reward", 0) or 0
    if rew != rew:
        rew = -50.0
    return 1000.0 * float(g) + float(rew)


def print_leaderboard(results, top_k=None):
    results = sorted(results, key=composite, reverse=True)
    if top_k:
        results = results[:top_k]
    if not results:
        print("(no completed trials yet)")
        return
    print(f"{'#':>3}  {'trial':<14}  {'gate':>4}  {'eval_R':>8}  "
          f"{'best_R':>8}  {'lr':>7}  {'ent':>7}  {'clip':>5}  "
          f"{'arch':<13}  {'act':<5}  {'time_s':>7}")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        hp = r.get("hparams", {})
        arch = "x".join(str(x) for x in hp.get("net_arch", []))
        print(f"{i:>3}  {r['_dir']:<14}  "
              f"{int(r.get('eval_max_gate_idx', 0)):>4}  "
              f"{r.get('eval_mean_reward', float('nan')):>8.2f}  "
              f"{r.get('best_mean_reward', float('nan')):>8.2f}  "
              f"{hp.get('learning_rate', 0):>7.0e}  "
              f"{hp.get('ent_coef', 0):>7.4f}  "
              f"{hp.get('clip_range', 0):>5.2f}  "
              f"{arch:<13}  "
              f"{hp.get('activation_fn', '?'):<5}  "
              f"{r.get('train_seconds', 0):>7.0f}")


def plot_optuna_diagnostics(study, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import optuna.visualization.matplotlib as ovm
    except ImportError as e:
        print(f"(skipping plots — missing dep: {e})")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("param_importance",
         lambda: ovm.plot_param_importances(study)),
        ("history",
         lambda: ovm.plot_optimization_history(study)),
        ("parallel",
         lambda: ovm.plot_parallel_coordinate(study)),
    ]
    import matplotlib.pyplot as plt
    for name, fn in plots:
        try:
            ax = fn()
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}.png", dpi=120)
            plt.close(fig)
            print(f"  wrote plots/{name}.png")
        except Exception as e:
            print(f"  ({name}) skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="ppo_hp_sweep")
    parser.add_argument(
        "--storage", type=str,
        default=f"sqlite:///{THIS_DIR / 'optuna_study.db'}")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results = load_results()
    print(f"\n=== Sweep leaderboard ({len(results)} completed trials) ===")
    print_leaderboard(results, top_k=args.top_k)

    try:
        study = optuna.load_study(study_name=args.study_name,
                                  storage=args.storage)
    except Exception as e:
        print(f"\n(could not load Optuna study: {e})")
        return

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]
    print(f"\nOptuna study: {len(completed)} complete, "
          f"{len(pruned)} pruned, {len(failed)} failed.")

    if completed:
        try:
            best = study.best_trial
            print(f"\nBest trial: #{best.number}  score={best.value:.2f}")
            print("Params:")
            for k, v in best.params.items():
                print(f"  {k} = {v}")
            best_path = THIS_DIR / "best_hparams.json"
            with open(best_path, "w") as f:
                json.dump({"trial": best.number, "score": best.value,
                           "params": best.params}, f, indent=2)
            print(f"Saved → {best_path}")
        except ValueError:
            print("\n(no best trial yet)")

    if not args.no_plots:
        plot_optuna_diagnostics(study, THIS_DIR / "plots")


if __name__ == "__main__":
    main()
