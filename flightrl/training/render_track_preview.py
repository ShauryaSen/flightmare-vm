#!/usr/bin/env python3
"""Render a 3D + 2D preview PNG for a track YAML in flightlib/configs/tracks/.

Wraps plot_track_layout.py's drawing logic but reads from a standalone track
file (rotation_quat OR rotation:[yaw,pitch,roll] in degrees) instead of the
main quadrotor_env.yaml. Outputs flightrl/training/track_previews/<name>.png.

Usage:
    python render_track_preview.py <track>          # one track
    python render_track_preview.py --all            # every yaml in tracks/
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.interpolate import CubicSpline
from ruamel.yaml import YAML

THIS_DIR = Path(__file__).resolve().parent
FLIGHTMARE_ROOT = THIS_DIR.parent.parent
TRACKS_DIR = FLIGHTMARE_ROOT / "flightlib" / "configs" / "tracks"
PREVIEW_DIR = THIS_DIR / "track_previews"
GATE_WIDTH = 5.0  # Match flightlib/configs/quadrotor_env.yaml `rl.gate_width`.

yaml = YAML()


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def euler_to_R(yaw_deg, pitch_deg, roll_deg):
    """Match the C++ env: R = Rz(yaw) * Ry(pitch) * Rx(roll), ZYX, degrees."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def gate_R(gate):
    if "rotation_quat" in gate:
        q = gate["rotation_quat"]
        return quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    if "rotation" in gate:
        r = gate["rotation"]
        return euler_to_R(float(r[0]), float(r[1]), float(r[2]))
    return np.eye(3)


def trackline(start_pos, gates, n_pts=300):
    waypoints = np.vstack([start_pos.reshape(1, 3), gates])
    t_knots = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        t_knots[i] = t_knots[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
    if t_knots[-1] == 0.0:
        return waypoints
    cs = CubicSpline(t_knots, waypoints, bc_type="natural")
    return cs(np.linspace(t_knots[0], t_knots[-1], n_pts))


def draw_gate_3d(ax, pos, R, idx, arrow_len=2.5):
    normal, right, up = R[:, 0], R[:, 1], R[:, 2]
    hw = GATE_WIDTH / 2
    corners = np.array([
        pos + hw * right + hw * up,
        pos - hw * right + hw * up,
        pos - hw * right - hw * up,
        pos + hw * right - hw * up,
        pos + hw * right + hw * up,
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            "b-", linewidth=2.2, alpha=0.85)
    ax.quiver(pos[0], pos[1], pos[2],
              normal[0] * arrow_len, normal[1] * arrow_len, normal[2] * arrow_len,
              color="red", linewidth=2.2, arrow_length_ratio=0.18)
    ax.quiver(pos[0], pos[1], pos[2],
              up[0] * arrow_len * 0.4, up[1] * arrow_len * 0.4, up[2] * arrow_len * 0.4,
              color="limegreen", linewidth=1.4, arrow_length_ratio=0.22, alpha=0.7)
    ax.text(pos[0], pos[1], pos[2] + hw + 0.3, f"G{idx}",
            fontsize=9, ha="center", color="blue", fontweight="bold")


def equal_aspect_3d(ax, all_pts):
    center = all_pts.mean(axis=0)
    extents = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3
    extents = max(extents, 5.0)
    ax.set_xlim(center[0] - extents, center[0] + extents)
    ax.set_ylim(center[1] - extents, center[1] + extents)
    ax.set_zlim(max(0, center[2] - extents), center[2] + extents)


def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


def render(track_name):
    track_path = TRACKS_DIR / f"{track_name}.yaml"
    if not track_path.exists():
        raise FileNotFoundError(track_path)
    with open(track_path) as f:
        cfg = yaml.load(f)

    start_pos = np.array(cfg["start_pos"], dtype=np.float64)
    gates_cfg = cfg["gates"]
    if not isinstance(gates_cfg, list):
        raise ValueError(f"{track_name}: gates must be a list")

    positions = np.array([list(g["position"]) for g in gates_cfg], dtype=np.float64)
    rotmats = [gate_R(g) for g in gates_cfg]
    description = cfg.get("description", "(no description)")
    n_gates = len(positions)

    pts = trackline(start_pos, positions)
    arc_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / f"{track_name}.png"

    # Composite figure: 3D on the left, top-down + side view on the right.
    fig = plt.figure(figsize=(20, 9))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")

    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
              "--", color="#FFD700", linewidth=2.2, alpha=0.7, label="spline")
    for i in range(n_gates):
        draw_gate_3d(ax3d, positions[i], rotmats[i], i)
    ax3d.scatter(*start_pos, color="green", s=200, marker="*", zorder=10,
                 edgecolors="black", linewidths=0.5, label="start")
    prev = start_pos
    for pos in positions:
        ax3d.plot([prev[0], pos[0]], [prev[1], pos[1]], [prev[2], pos[2]],
                  "-", color="gray", linewidth=0.7, alpha=0.4)
        prev = pos
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D")
    ax3d.legend(fontsize=8, loc="upper left")
    equal_aspect_3d(ax3d, np.vstack([positions, start_pos.reshape(1, 3)]))
    ax3d.view_init(elev=30, azim=-60)

    # 2D panels: XY top-down + XZ side. Slots 2 and 3 in the 1x3 grid.
    panels = [
        (2, 0, 1, "X (m)", "Y (m)", "Top-down (XY)"),
        (3, 0, 2, "X (m)", "Z (m)", "Side (XZ)"),
    ]
    for slot, dim_a, dim_b, xlabel, ylabel, title in panels:
        ax = fig.add_subplot(1, 3, slot)
        ax.plot(pts[:, dim_a], pts[:, dim_b],
                "--", color="#FFD700", linewidth=2, alpha=0.7, label="spline")
        for i in range(n_gates):
            pos = positions[i]
            R = rotmats[i]
            normal = R[:, 0]
            hw = GATE_WIDTH / 2
            # Gate opening line: perpendicular to projected normal in this plane.
            n2d = np.array([normal[dim_a], normal[dim_b]])
            n2d_norm = np.linalg.norm(n2d)
            if n2d_norm < 1e-6:
                # Normal is perpendicular to this plane; show a small dot only.
                ax.plot(pos[dim_a], pos[dim_b], "bs", markersize=6)
            else:
                perp = np.array([-n2d[1], n2d[0]]) / n2d_norm
                p1 = np.array([pos[dim_a], pos[dim_b]]) + perp * hw
                p2 = np.array([pos[dim_a], pos[dim_b]]) - perp * hw
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                        color="blue", linewidth=2.5, alpha=0.85, zorder=4)
            ax.plot(pos[dim_a], pos[dim_b], "bD", markersize=6, zorder=5)

            # Normal arrow (red).
            ax.annotate("",
                        xy=(pos[dim_a] + normal[dim_a] * 2.5,
                            pos[dim_b] + normal[dim_b] * 2.5),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle="->", color="red",
                                        lw=1.8, mutation_scale=14), zorder=6)

            # Segment-direction arrow (cyan): the direction the spline enters at.
            g_prev = start_pos if i == 0 else positions[i - 1]
            seg = positions[i] - g_prev
            seg_len = np.linalg.norm(seg)
            if seg_len > 1e-6:
                seg = seg / seg_len
                ax.annotate("",
                            xy=(pos[dim_a] + seg[dim_a] * 2.0,
                                pos[dim_b] + seg[dim_b] * 2.0),
                            xytext=(pos[dim_a], pos[dim_b]),
                            arrowprops=dict(arrowstyle="->", color="cyan",
                                            lw=1.4, mutation_scale=12,
                                            alpha=0.7), zorder=5)

            # Misalignment label between gate normal and segment-entering direction.
            misalign = angle_between(normal, seg)
            color = "green" if misalign < 20 else ("orange" if misalign < 45 else "red")
            ax.annotate(f"G{i} ({misalign:.0f}°)",
                        (pos[dim_a], pos[dim_b]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=color, fontweight="bold")
        ax.scatter(start_pos[dim_a], start_pos[dim_b],
                   color="green", s=120, marker="*", zorder=10, label="start")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{track_name} — {description}\n"
        f"{n_gates} gates · approx spline length {arc_length:.1f} m · "
        f"red=normal, cyan=segment-entering dir, label=misalignment angle",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_gates, arc_length


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("track", nargs="?", help="track name (without .yaml)")
    p.add_argument("--all", action="store_true",
                   help="render every track in flightlib/configs/tracks/")
    args = p.parse_args()

    if args.all:
        names = sorted(t.stem for t in TRACKS_DIR.glob("*.yaml"))
    elif args.track:
        names = [args.track]
    else:
        p.error("specify a track name or --all")

    for name in names:
        try:
            out, n, arc = render(name)
            rel = out.relative_to(FLIGHTMARE_ROOT)
            print(f"  ✓ {name:<14} {n:>2} gates  {arc:>6.1f} m   {rel}")
        except Exception as e:
            print(f"  ✗ {name:<14} FAILED: {e}")


if __name__ == "__main__":
    main()
