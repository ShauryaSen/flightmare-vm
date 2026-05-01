#!/usr/bin/env python3
"""Zoomed visualization of the G0->G1->G2->G3 semicircle arc."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import ruamel.yaml
import os


def quat_to_R(w, x, y, z):
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def hermite_spline(p0, p1, t0, t1, n=80):
    chord = np.linalg.norm(p1 - p0)
    m0 = t0 * chord
    m1 = t1 * chord
    s = np.linspace(0, 1, n)[:, None]
    s2 = s * s
    s3 = s2 * s
    h00 = 2*s3 - 3*s2 + 1
    h10 = s3 - 2*s2 + s
    h01 = -2*s3 + 3*s2
    h11 = s3 - s2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1


def main():
    cfg_path = os.path.expanduser("~/flightmare/flightlib/configs/quadrotor_env.yaml")
    yaml = ruamel.yaml.YAML()
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    start_pos = np.array(cfg["start_pos"], dtype=float)
    gates_cfg = cfg["gates"]
    positions = np.array([g["position"] for g in gates_cfg], dtype=float)
    rotmats = [quat_to_R(*[float(x) for x in g["rotation_quat"]]) for g in gates_cfg]
    normals = np.array([R[:, 0] for R in rotmats])
    gate_width = float(cfg["rl"]["gate_width"])
    half = gate_width / 2

    ctrl = np.vstack([start_pos[None, :], positions])
    n_ctrl = len(ctrl)
    tangents = np.zeros_like(ctrl)
    tangents[0] = (positions[0] - start_pos)
    tangents[0] /= np.linalg.norm(tangents[0])
    for i in range(len(positions)):
        tangents[i + 1] = normals[i]

    spline_pts = []
    for i in range(n_ctrl - 1):
        seg = hermite_spline(ctrl[i], ctrl[i + 1], tangents[i], tangents[i + 1], n=80)
        spline_pts.append(seg if i == 0 else seg[1:])
    spline_pts = np.vstack(spline_pts)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Panel 1: zoomed top-down on G0..G3 ──
    ax = axes[0]
    n_focus = 4  # G0, G1, G2, G3
    focus_pts = positions[:n_focus]

    # ideal arc circle (chord G0-G3 in xy)
    g0xy = positions[0, :2]
    g3xy = positions[3, :2]
    C = (g0xy + g3xy) / 2
    R = np.linalg.norm(g0xy - C)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(C[0] + R*np.cos(theta), C[1] + R*np.sin(theta),
            ':', color='gray', alpha=0.5, lw=1.5, label=f'Ideal circle (R={R:.2f}m)')

    # semicircle arc highlighted
    a0 = np.arctan2(g0xy[1] - C[1], g0xy[0] - C[0])
    a3 = np.arctan2(g3xy[1] - C[1], g3xy[0] - C[0])
    if a3 < a0:
        a3 += 2*np.pi
    arc_theta = np.linspace(a0, a3, 100)
    ax.plot(C[0] + R*np.cos(arc_theta), C[1] + R*np.sin(arc_theta),
            '-', color='orange', alpha=0.6, lw=3, label='Target semicircle')

    # actual hermite spline G0..G3
    seg_idx_start = 0
    seg_idx_end = 3 * 80 - (3 - 1)  # rough; just plot all
    ax.plot(spline_pts[:n_focus*80, 0], spline_pts[:n_focus*80, 1],
            '--', color='gold', lw=2.5, alpha=0.9, label='Hermite spline (actual)')

    # gates
    colors = ['#1f77b4', '#d62728', '#d62728', '#1f77b4']  # blue for G0/G3, red for redesigned G1/G2
    labels = ['G0 (anchor)', 'G1 (on arc)', 'G2 (on arc)', 'G3 (anchor)']
    for i in range(n_focus):
        pos = positions[i]
        n = normals[i]
        right = rotmats[i][:, 1]
        # gate opening line in xy
        gx = np.array([pos[0] - right[0]*half, pos[0] + right[0]*half])
        gy = np.array([pos[1] - right[1]*half, pos[1] + right[1]*half])
        ax.plot(gx, gy, '-', color=colors[i], lw=4, solid_capstyle='round')
        ax.plot(pos[0], pos[1], 'o', color=colors[i], ms=10, mec='black', mew=0.5, zorder=5)
        # normal arrow
        ax.annotate('', xy=(pos[0] + n[0]*3.5, pos[1] + n[1]*3.5),
                    xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.0, mutation_scale=15))
        ax.text(pos[0], pos[1] + 1.2, f"G{i}", fontsize=12, fontweight='bold',
                ha='center', color=colors[i])

    ax.scatter(*start_pos[:2], color='green', s=200, marker='*', zorder=6,
               edgecolors='black', lw=0.5, label='Start')
    ax.scatter(*C, marker='+', color='gray', s=150, zorder=4)
    ax.text(C[0], C[1] - 1, f'C=({C[0]:.1f},{C[1]:.1f})', ha='center', fontsize=9, color='gray')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top-Down (XY): G0 → G1 → G2 → G3 Semicircle',
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # zoom around the four gates
    pad = 4
    xs = focus_pts[:, 0]
    ys = focus_pts[:, 1]
    ax.set_xlim(min(xs.min(), start_pos[0]) - pad, xs.max() + pad)
    ax.set_ylim(min(ys.min(), start_pos[1]) - pad, ys.max() + pad)

    # ── Panel 2: side view (XZ) ──
    ax2 = axes[1]
    ax2.plot(spline_pts[:n_focus*80, 0], spline_pts[:n_focus*80, 2],
             '--', color='gold', lw=2.5, alpha=0.9, label='Hermite spline')
    for i in range(n_focus):
        pos = positions[i]
        up = rotmats[i][:, 2]
        # gate opening line in xz: use the up vector projected to xz
        # actually for side view, plot a vertical bar showing gate height
        ax2.plot([pos[0], pos[0]], [pos[2] - half, pos[2] + half],
                 '-', color=colors[i], lw=4, solid_capstyle='round')
        ax2.plot(pos[0], pos[2], 'o', color=colors[i], ms=10, mec='black', mew=0.5, zorder=5)
        n = normals[i]
        ax2.annotate('', xy=(pos[0] + n[0]*3.5, pos[2] + n[2]*3.5),
                     xytext=(pos[0], pos[2]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2.0, mutation_scale=15))
        ax2.text(pos[0], pos[2] + half + 0.5, f"G{i}", fontsize=12, fontweight='bold',
                 ha='center', color=colors[i])
    ax2.scatter(start_pos[0], start_pos[2], color='green', s=200, marker='*', zorder=6,
                edgecolors='black', lw=0.5, label='Start')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title('Side (XZ): altitude profile through arc',
                  fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(min(xs.min(), start_pos[0]) - pad, xs.max() + pad)

    fig.suptitle('Semicircle redesign: G1=(15.81,11.49,4.87) at 60°, G2=(8.28,14.18,3.43) at 120°',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.expanduser("~/flightmare/flightrl/training/semicircle_check.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
