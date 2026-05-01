#!/usr/bin/env python3
"""Morning analysis script.

Reads all sweep/runs/<condition>/result.json + progress.json and produces:
  - ranked terminal table (sorted by eval_max_gate_idx, then eval_mean_reward)
  - sweep/comparison.png  (all conditions overlaid, top-3 highlighted)
  - sweep/per_condition.png  (grid of per-condition curves)
  - sweep/SUMMARY.md  (markdown ranked table + interpretation)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"


def load_runs():
    runs = []
    if not RUNS_DIR.exists():
        sys.exit(f"FATAL: {RUNS_DIR} does not exist — has the sweep run yet?")
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        result_path = d / "result.json"
        progress_path = d / "progress.json"
        cond_path = d / "condition.json"
        if not result_path.exists():
            print(f"  [skip] {d.name}: no result.json (training may have failed)")
            continue
        with open(result_path) as f:
            result = json.load(f)
        progress = []
        if progress_path.exists():
            with open(progress_path) as f:
                progress = json.load(f)
        condition = {}
        if cond_path.exists():
            with open(cond_path) as f:
                condition = json.load(f)
        runs.append({
            "name": d.name,
            "result": result,
            "progress": progress,
            "condition": condition,
        })
    return runs


def rank_runs(runs):
    """Sort by eval_max_gate_idx desc, then eval_mean_reward desc."""

    def key(r):
        res = r["result"]
        return (
            -float(res.get("eval_max_gate_idx", 0) or 0),
            -float(res.get("eval_mean_reward", -1e9) or -1e9),
            -float(res.get("final_ep_rew_mean", -1e9) or -1e9),
        )

    return sorted(runs, key=key)


def fmt_num(v, nd=2):
    if v is None:
        return "—"
    try:
        if isinstance(v, bool):
            return "yes" if v else "no"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def print_table(runs):
    print()
    print("=" * 110)
    header = (f"{'rank':>4}  {'condition':<26}  {'eval_gate':>9}  {'eval_rew':>9}  "
              f"{'eval_len':>9}  {'lap?':>5}  {'final_rew':>9}  {'slope/M':>8}  {'min':>5}")
    print(header)
    print("-" * 110)
    for i, r in enumerate(runs):
        res = r["result"]
        line = (f"{i+1:>4}  {r['name']:<26}  "
                f"{fmt_num(res.get('eval_max_gate_idx'), 1):>9}  "
                f"{fmt_num(res.get('eval_mean_reward')):>9}  "
                f"{fmt_num(res.get('eval_mean_length'), 0):>9}  "
                f"{fmt_num(res.get('eval_lap_completed')):>5}  "
                f"{fmt_num(res.get('final_ep_rew_mean')):>9}  "
                f"{fmt_num(res.get('slope_last_2M'), 2):>8}  "
                f"{fmt_num((res.get('train_seconds') or 0) / 60, 1):>5}")
        print(line)
    print("=" * 110)


def plot_comparison(runs, out_path):
    """All conditions overlaid; top-3 in color, rest in gray."""
    ranked = rank_runs(runs)
    top3_names = {r["name"] for r in ranked[:3]}

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    color_map = {r["name"]: colors[i] for i, r in enumerate(ranked[:3])}

    for r in runs:
        progress = r["progress"]
        if not progress:
            continue
        steps = [p["step"] / 1e6 for p in progress]
        rew = [p["ep_rew_mean"] for p in progress]
        if r["name"] in top3_names:
            ax.plot(steps, rew, label=f"{r['name']} (rank {ranked.index(r)+1})",
                    color=color_map[r["name"]], linewidth=2.2, zorder=3)
        else:
            ax.plot(steps, rew, color="#bbbbbb", linewidth=0.9, zorder=1, alpha=0.7)

    ax.set_xlabel("Million training steps")
    ax.set_ylabel("ep_rew_mean (last 100 episodes)")
    ax.set_title("Sweep comparison — top 3 highlighted")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_condition(runs, out_path):
    n = len(runs)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.7 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, r in enumerate(runs):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        progress = r["progress"]
        if progress:
            steps = [p["step"] / 1e6 for p in progress]
            rew = [p["ep_rew_mean"] for p in progress]
            length = [p["ep_len_mean"] for p in progress]
            ax.plot(steps, rew, color="#1f77b4", linewidth=1.5, label="rew")
            ax2 = ax.twinx()
            ax2.plot(steps, length, color="#ff7f0e", linewidth=1.0,
                     alpha=0.6, label="len")
            ax2.set_ylim(0, 1600)
            ax2.tick_params(axis="y", labelsize=7)
        res = r["result"]
        gate = res.get("eval_max_gate_idx", 0)
        ax.set_title(f"{r['name']}\nmax_gate={fmt_num(gate, 1)}, "
                     f"rew={fmt_num(res.get('final_ep_rew_mean'))}",
                     fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-condition training curves "
                 "(blue = ep_rew_mean, orange = ep_len_mean)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_md(runs, out_path):
    ranked = rank_runs(runs)
    lines = [
        "# Sweep Results",
        "",
        f"**Conditions:** {len(runs)}",
        "",
        "## Ranked by eval_max_gate_idx → eval_mean_reward",
        "",
        "| rank | condition | eval_gate | eval_rew | eval_len | lap? | "
        "final_rew | slope/M | min |",
        "|------|-----------|-----------|----------|----------|------|-----------|---------|-----|",
    ]
    for i, r in enumerate(ranked):
        res = r["result"]
        lines.append(
            f"| {i+1} | `{r['name']}` | "
            f"{fmt_num(res.get('eval_max_gate_idx'), 1)} | "
            f"{fmt_num(res.get('eval_mean_reward'))} | "
            f"{fmt_num(res.get('eval_mean_length'), 0)} | "
            f"{fmt_num(res.get('eval_lap_completed'))} | "
            f"{fmt_num(res.get('final_ep_rew_mean'))} | "
            f"{fmt_num(res.get('slope_last_2M'), 2)} | "
            f"{fmt_num((res.get('train_seconds') or 0) / 60, 1)} |"
        )

    # Interpretation
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if ranked:
        winner = ranked[0]
        wname = winner["name"]
        wgate = winner["result"].get("eval_max_gate_idx", 0) or 0
        runner_up_gate = ranked[1]["result"].get("eval_max_gate_idx", 0) if len(ranked) > 1 else 0

        if wgate <= 2:
            lines.append(
                f"- **No condition learned to clear gates reliably** (best max_gate={wgate}). "
                "This suggests a deeper bug exists (reward shape, sim timing, init distribution). "
                "Hyperparameter sweep won't fix this — investigate before more training."
            )
        elif wgate - (runner_up_gate or 0) >= 2:
            lines.append(
                f"- **Clear winner: `{wname}`** (max_gate={wgate}, runner-up={runner_up_gate}). "
                "Re-run with 5 seeds at 25M steps for a paper-quality result."
            )
        else:
            top = [r["name"] for r in ranked[:3]]
            lines.append(
                f"- **Tied top tier: {top}**. Top 3 are within ~1 gate of each other; "
                "could be seed noise. Re-run those with 3 seeds at 40M to break the tie."
            )

        # Steep-slope dark horses
        steep = [r for r in ranked if (r["result"].get("slope_last_2M") or 0) > 1.0]
        if steep:
            names = [r["name"] for r in steep[:5]]
            lines.append(
                f"- **Steep-slope dark horses (slope/M > 1.0):** {names}. "
                "These may not have converged — promote to the finalist re-run."
            )

        # Health checks
        broken = [r["name"] for r in runs
                  if (r["result"].get("final_ep_len_mean") or 0) < 100]
        if broken:
            lines.append(
                f"- **Likely-broken runs (ep_len_mean < 100):** {broken}. "
                "Drone is crashing fast — config may be reward-hacking or destabilizing."
            )

    lines += [
        "",
        "## Decision rules used",
        "",
        "1. Primary: `eval_max_gate_idx` (deterministic eval from start_pos)",
        "2. Tiebreaker 1: `eval_lap_completed`",
        "3. Tiebreaker 2: `eval_mean_reward`",
        "4. Watch: `slope_last_2M` for steep-slope candidates that needed more training",
        "5. Sanity: `ep_len_mean` (1500 = full episode, ~90 = crashing fast)",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    runs = load_runs()
    if not runs:
        sys.exit("No completed runs found.")

    ranked = rank_runs(runs)
    print_table(ranked)

    print("\nWriting plots and summary…")
    plot_comparison(runs, THIS_DIR / "comparison.png")
    plot_per_condition(runs, THIS_DIR / "per_condition.png")
    write_summary_md(ranked, THIS_DIR / "SUMMARY.md")

    print("\nArtifacts:")
    print(f"  {THIS_DIR / 'SUMMARY.md'}")
    print(f"  {THIS_DIR / 'comparison.png'}")
    print(f"  {THIS_DIR / 'per_condition.png'}")


if __name__ == "__main__":
    main()
