#!/usr/bin/env python3
"""Visualize Condition C tracks across the complexity schedule.

Renders what the per-episode randomized track actually looks like at each
complexity c the AdaptiveComplexity controller may settle at over training,
drawing the EXACT same cubic Hermite spline the C++ env uses for the
progress reward (see quadrotor_env.cpp::buildTrackSpline).

Output: one PNG per c plus a grid summary, top-down (XY) and 3D, with
N_DRAWS independent samples overlaid per c so you can see the *distribution*
of tracks the policy will train on at that complexity level.

Usage:
    python visualize_random_tracks.py [--out-dir DIR] [--n-draws N] [--seed S]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from track_generator import TrackGenerator

# Snapshots along the curriculum lifetime.
COMPLEXITY_SCHEDULE = [
    ("c=0.00 (baseline straight line)", 0.00),
    ("c=0.10 (early curriculum)",        0.10),
    ("c=0.30",                           0.30),
    ("c=0.50",                           0.50),
    ("c=0.70",                           0.70),
    ("c=1.00 (max complexity)",          1.00),
]
N_DRAWS = 6
SAMPLES_PER_SEGMENT = 200      # mirror kSplineSamplesPerSegment
GATE_HALF_WIDTH = 2.5          # gate_width=5m → half = 2.5m
GATE_HEIGHT_HALF = 2.5


def hermite_eval(p0, m0, p1, m1, t):
    """Cubic Hermite — vectorised over t."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (h00[:, None] * p0 + h10[:, None] * m0
            + h01[:, None] * p1 + h11[:, None] * m1)


def build_spline(start_pos, gates_pos, gate_yaws_rad,
                 samples_per_segment=SAMPLES_PER_SEGMENT):
    """Replicate QuadrotorEnv::buildTrackSpline exactly."""
    num_gates = len(gates_pos)
    ctrl_pts = np.vstack([start_pos[None, :], gates_pos])
    tangent_dirs = np.zeros_like(ctrl_pts)
    sd = gates_pos[0] - start_pos
    if np.linalg.norm(sd) > 1e-6:
        tangent_dirs[0] = sd / np.linalg.norm(sd)
    for i in range(num_gates):
        tangent_dirs[i + 1] = (np.cos(gate_yaws_rad[i]),
                               np.sin(gate_yaws_rad[i]), 0.0)

    out = []
    t = np.linspace(0.0, 1.0, samples_per_segment, dtype=np.float32)
    for seg_i in range(num_gates):
        p0 = ctrl_pts[seg_i]
        p1 = ctrl_pts[seg_i + 1]
        chord = np.linalg.norm(p1 - p0)
        m0 = tangent_dirs[seg_i] * chord
        m1 = tangent_dirs[seg_i + 1] * chord
        seg = hermite_eval(p0, m0, p1, m1, t)
        if seg_i > 0:
            seg = seg[1:]
        out.append(seg)
    return np.concatenate(out, axis=0)


def gate_endpoints_xy(center, yaw_rad, half=GATE_HALF_WIDTH):
    rx, ry = -np.sin(yaw_rad), np.cos(yaw_rad)
    p_left  = (center[0] - rx * half, center[1] - ry * half)
    p_right = (center[0] + rx * half, center[1] + ry * half)
    return p_left, p_right


def gate_normal_arrow_xy(center, yaw_rad, length=2.0):
    fx, fy = np.cos(yaw_rad), np.sin(yaw_rad)
    return (center[0], center[1], length * fx, length * fy)


def draw_track_2d(ax, start_pos, gates_pos, yaws, color, alpha,
                  show_gates=True, show_arrows=False, gate_lw=2.0,
                  spline_lw=1.5, label=None):
    spline = build_spline(start_pos, gates_pos, yaws)
    ax.plot(spline[:, 0], spline[:, 1], color=color, alpha=alpha,
            lw=spline_lw, label=label)
    if show_gates:
        for c, y in zip(gates_pos, yaws):
            (x0, y0), (x1, y1) = gate_endpoints_xy(c, y)
            ax.plot([x0, x1], [y0, y1], color=color,
                    alpha=min(1.0, alpha + 0.2), lw=gate_lw)
    if show_arrows:
        for c, y in zip(gates_pos, yaws):
            ax.arrow(*gate_normal_arrow_xy(c, y),
                     head_width=0.6, head_length=0.8, fc=color, ec=color,
                     alpha=alpha, length_includes_head=True)


def draw_track_3d(ax, start_pos, gates_pos, yaws, color, alpha,
                  show_gates=True, spline_lw=1.5):
    spline = build_spline(start_pos, gates_pos, yaws)
    ax.plot(spline[:, 0], spline[:, 1], spline[:, 2],
            color=color, alpha=alpha, lw=spline_lw)
    if show_gates:
        for c, y in zip(gates_pos, yaws):
            rx, ry = -np.sin(y), np.cos(y)
            p_left  = c + np.array([rx, ry, 0]) * GATE_HALF_WIDTH
            p_right = c - np.array([rx, ry, 0]) * GATE_HALF_WIDTH
            top_off = np.array([0, 0, GATE_HEIGHT_HALF])
            corners = np.array([p_left + top_off, p_right + top_off,
                                p_right - top_off, p_left - top_off,
                                p_left + top_off])
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
                    color=color, alpha=min(1.0, alpha + 0.15), lw=2)


def world_extents(generator, complexities, n_draws, seed=999):
    """Quick bounding-box probe so all panels share the same axes."""
    g = TrackGenerator(num_gates=generator.num_gates,
                       forward_m=generator.forward,
                       lateral_max_m=generator.lateral_max,
                       vertical_max_m=generator.vertical_max,
                       yaw_max_deg=np.rad2deg(generator.yaw_max),
                       start_pos=tuple(generator.start_pos.tolist()),
                       seed=seed)
    pts = [generator.start_pos[None, :]]
    for c in complexities:
        for _ in range(n_draws):
            p, _ = g.sample(c)
            pts.append(p)
    P = np.vstack(pts)
    pad = 3.0
    return (P[:, 0].min() - pad, P[:, 0].max() + pad,
            P[:, 1].min() - pad, P[:, 1].max() + pad,
            P[:, 2].min() - pad, P[:, 2].max() + pad)


def render_per_complexity(generator: TrackGenerator, label: str, c: float,
                          out_dir: Path, n_draws: int,
                          extents):
    fig = plt.figure(figsize=(13, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    sp = generator.start_pos

    # Baseline (gold) underneath every draw.
    bp, by = generator.baseline()
    draw_track_2d(ax2d, sp, bp, by, color="goldenrod", alpha=0.40,
                  show_gates=True, show_arrows=True, gate_lw=2.5,
                  spline_lw=2.0, label="baseline (c=0)")
    draw_track_3d(ax3d, sp, bp, by, color="goldenrod", alpha=0.40,
                  show_gates=True, spline_lw=2.0)

    palette = plt.cm.viridis(np.linspace(0.15, 0.95, n_draws))
    for k in range(n_draws):
        pos, yaw = generator.sample(c)
        col = palette[k]
        draw_track_2d(ax2d, sp, pos, yaw, color=col, alpha=0.85,
                      show_gates=True, show_arrows=False, gate_lw=1.5,
                      spline_lw=1.2,
                      label=(f"draw {k+1}" if c > 0 else None))
        draw_track_3d(ax3d, sp, pos, yaw, color=col, alpha=0.85,
                      show_gates=True, spline_lw=1.2)

    ax2d.scatter([sp[0]], [sp[1]], marker="*", s=160, color="red",
                 zorder=10, label="start_pos")
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("x [m]"); ax2d.set_ylabel("y [m]")
    ax2d.grid(alpha=0.3)
    ax2d.set_title(f"top-down — {label}")
    ax2d.legend(loc="upper right", fontsize=8, framealpha=0.85)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = extents
    ax2d.set_xlim(x_lo, x_hi); ax2d.set_ylim(y_lo, y_hi)

    ax3d.scatter([sp[0]], [sp[1]], [sp[2]], marker="*", s=80, color="red")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"3D — {label}")
    ax3d.set_xlim(x_lo, x_hi); ax3d.set_ylim(y_lo, y_hi); ax3d.set_zlim(z_lo, z_hi)
    ax3d.view_init(elev=25, azim=-65)

    fig.suptitle(f"Condition C — {n_draws} draws | {label}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"complexity_{c:0.2f}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_overview_grid(generator: TrackGenerator, out_dir: Path,
                         n_draws: int, extents):
    n = len(COMPLEXITY_SCHEDULE)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows),
                             squeeze=False)
    sp = generator.start_pos
    bp, by = generator.baseline()
    palette = plt.cm.viridis(np.linspace(0.15, 0.95, n_draws))
    x_lo, x_hi, y_lo, y_hi, *_ = extents

    for idx, (label, c) in enumerate(COMPLEXITY_SCHEDULE):
        ax = axes[idx // cols][idx % cols]
        draw_track_2d(ax, sp, bp, by, color="goldenrod", alpha=0.40,
                      show_gates=True, show_arrows=False,
                      gate_lw=2.0, spline_lw=2.0)
        for k in range(n_draws):
            pos, yaw = generator.sample(c)
            draw_track_2d(ax, sp, pos, yaw, color=palette[k], alpha=0.75,
                          show_gates=True, show_arrows=False,
                          gate_lw=1.0, spline_lw=1.0)
        ax.scatter([sp[0]], [sp[1]], marker="*", s=120, color="red", zorder=10)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(f"Condition C — random-track curriculum across complexity "
                 f"({n_draws} draws/c) — start: straight line, end: full variability",
                 fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "_overview_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-draws", type=int, default=N_DRAWS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    flight_root = here.parents[2]
    yaml_path = flight_root / "flightlib" / "configs" / "tracks" / "linear_8.yaml"

    out_dir = Path(args.out_dir) if args.out_dir else \
        here / "track_previews_random"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = TrackGenerator.from_baseline_yaml(yaml_path, seed=args.seed)
    print(f"baseline linear_8: {gen.num_gates} gates, "
          f"forward={gen.forward}m, max_lat={gen.lateral_max}m, "
          f"max_vert={gen.vertical_max}m, max_yaw={np.rad2deg(gen.yaw_max):.1f}°")

    cs = [c for _, c in COMPLEXITY_SCHEDULE]
    extents = world_extents(gen, cs, args.n_draws, seed=args.seed + 999)

    paths = []
    for label, c in COMPLEXITY_SCHEDULE:
        # Reset RNG per panel so the first row's tracks are reproducible.
        gen.rng = np.random.default_rng(args.seed + int(c * 1000))
        p = render_per_complexity(gen, label, c, out_dir, args.n_draws,
                                  extents)
        paths.append(p)
        print(f"  wrote {p}  (c={c})")

    gen.rng = np.random.default_rng(args.seed + 12345)
    grid = render_overview_grid(gen, out_dir, args.n_draws, extents)
    print(f"  wrote {grid}")

    print(f"\nDone — {len(paths) + 1} files in {out_dir}")


if __name__ == "__main__":
    main()
