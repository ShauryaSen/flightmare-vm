#!/usr/bin/env python3
"""
Generate a 3D track layout visualization showing gates with corrected normals.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import ruamel.yaml
import os


def quat_to_R(w, x, y, z):
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def compute_trackline(start_pos, gates, n_pts=300):
    waypoints = np.vstack([start_pos.reshape(1, 3), gates])
    t_knots = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        t_knots[i] = t_knots[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
    cs = CubicSpline(t_knots, waypoints, bc_type='natural')
    t_fine = np.linspace(t_knots[0], t_knots[-1], n_pts)
    return cs(t_fine)


def draw_gate_3d(ax, pos, R, gate_width, idx, arrow_len=3.0):
    normal = R[:, 0]
    right = R[:, 1]
    up = R[:, 2]
    hw = gate_width / 2

    corners = np.array([
        pos + hw * right + hw * up,
        pos - hw * right + hw * up,
        pos - hw * right - hw * up,
        pos + hw * right - hw * up,
        pos + hw * right + hw * up,
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            "b-", linewidth=2.5, alpha=0.85)

    ax.quiver(pos[0], pos[1], pos[2],
              normal[0]*arrow_len, normal[1]*arrow_len, normal[2]*arrow_len,
              color='red', linewidth=2.5, arrow_length_ratio=0.15)
    ax.quiver(pos[0], pos[1], pos[2],
              up[0]*arrow_len*0.4, up[1]*arrow_len*0.4, up[2]*arrow_len*0.4,
              color='limegreen', linewidth=1.5, arrow_length_ratio=0.2, alpha=0.7)

    ax.text(pos[0], pos[1], pos[2] + hw + 0.3, f"G{idx}",
            fontsize=9, ha="center", color="blue", fontweight="bold")


def main():
    config_path = os.path.expanduser("~/flightmare/flightlib/configs/quadrotor_env.yaml")
    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)

    start_pos = np.array(cfg["start_pos"], dtype=np.float64)
    gates_cfg = cfg["gates"]

    positions = []
    rotmats = []
    for g in gates_cfg:
        pos = np.array(g["position"], dtype=np.float64)
        positions.append(pos)
        q = g["rotation_quat"]
        R = quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        rotmats.append(R)

    positions = np.array(positions)
    gate_width = float(cfg["rl"]["gate_width"])

    # ── 3D Plot ──
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Trackline
    pts = compute_trackline(start_pos, positions)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            '--', color='#FFD700', linewidth=2.5, alpha=0.7, label='Center Trackline')

    # Gates
    for i in range(len(positions)):
        draw_gate_3d(ax, positions[i], rotmats[i], gate_width, i, arrow_len=3.0)

    # Start position
    ax.scatter(*start_pos, color="green", s=200, marker="*", zorder=10,
               edgecolors="black", linewidths=0.5, label="Start Position")

    # Segment lines (thin gray)
    prev = start_pos
    for i, pos in enumerate(positions):
        ax.plot([prev[0], pos[0]], [prev[1], pos[1]], [prev[2], pos[2]],
                '-', color='gray', linewidth=0.8, alpha=0.4)
        prev = pos

    # Formatting
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title("Gate Track Layout — Normals Aligned to Segment Direction\n"
                 "(Red arrows = gate normal / fly-through direction)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc='upper left')

    # Equal aspect
    all_pts = np.vstack([positions, start_pos.reshape(1, 3)])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(max(0, center[2] - max_range), center[2] + max_range)

    # Set viewing angle similar to user's original screenshot
    ax.view_init(elev=35, azim=-60)

    out_path = os.path.expanduser("~/flightmare/flightrl/training/track_layout_fixed.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")

    # ── Also create a top-down (XY) view ──
    fig2, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax_idx, (ax, dim_a, dim_b, xlabel, ylabel, title) in enumerate([
        (axes[0], 0, 1, "X (m)", "Y (m)", "Top-Down (XY) — Normals Along Track"),
        (axes[1], 0, 2, "X (m)", "Z (m)", "Side View (XZ) — Normals Along Track"),
    ]):
        # Trackline
        ax.plot(pts[:, dim_a], pts[:, dim_b],
                '--', color='#FFD700', linewidth=2, alpha=0.7, label='Trackline')

        for i in range(len(positions)):
            pos = positions[i]
            R = rotmats[i]
            normal = R[:, 0]
            right = R[:, 1]

            # Gate opening bar
            hw = gate_width / 2
            gate_perp_2d = np.array([-normal[dim_b], normal[dim_a]])
            if np.linalg.norm(gate_perp_2d) > 1e-6:
                gate_perp_2d /= np.linalg.norm(gate_perp_2d)
            p1 = np.array([pos[dim_a], pos[dim_b]]) + gate_perp_2d * hw
            p2 = np.array([pos[dim_a], pos[dim_b]]) - gate_perp_2d * hw
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color='blue', linewidth=3, alpha=0.8, zorder=4)
            ax.plot(pos[dim_a], pos[dim_b], "bD", markersize=8, zorder=5)

            # Gate NORMAL arrow
            arrow_len = 3.0
            ax.annotate('', xy=(pos[dim_a] + normal[dim_a]*arrow_len,
                                pos[dim_b] + normal[dim_b]*arrow_len),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                       lw=2.0, mutation_scale=15), zorder=6)

            # Segment direction arrow
            if i == 0:
                g_prev = start_pos
            else:
                g_prev = positions[i-1]
            seg_dir = positions[i] - g_prev
            seg_norm = np.linalg.norm(seg_dir)
            if seg_norm > 1e-6:
                seg_dir = seg_dir / seg_norm

            ax.annotate('', xy=(pos[dim_a] + seg_dir[dim_a]*arrow_len*0.7,
                                pos[dim_b] + seg_dir[dim_b]*arrow_len*0.7),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='cyan',
                                       lw=1.5, mutation_scale=12, alpha=0.7), zorder=5)

            # Angle label
            dot_val = np.dot(normal[:3], seg_dir[:3])
            angle = np.degrees(np.arccos(np.clip(abs(dot_val), 0, 1)))
            label_color = 'lime' if angle < 20 else ('yellow' if angle < 45 else 'red')
            ax.annotate(f"G{i} ({angle:.0f}°)", (pos[dim_a], pos[dim_b]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color=label_color, fontweight="bold")

        ax.scatter(start_pos[dim_a], start_pos[dim_b], color="green",
                   s=120, marker="*", zorder=10, label="Start")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig2.suptitle("Fixed Gate Orientations — All Normals Aligned to Segment Direction (0° error)",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path2 = os.path.expanduser("~/flightmare/flightrl/training/track_layout_fixed_2d.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Saved: {out_path2}")


if __name__ == "__main__":
    main()
