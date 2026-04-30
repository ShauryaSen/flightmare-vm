#!/usr/bin/env python3
"""
Fix gate rotations using spline tangent directions.

Fits a cubic spline through [start_pos, gate0, gate1, ..., gateN] and uses
the spline derivative at each gate to determine the correct fly-through
direction.  This avoids the "kinked" headings that arise from using raw
segment vectors (prev → current) and gives a smooth, natural racing line.

Usage:
    python fix_gate_rotations_spline.py            # dry-run (report only)
    python fix_gate_rotations_spline.py --apply     # write to YAML
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import ruamel.yaml
import os


# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────

def quat_to_R(w, x, y, z):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def direction_to_quat(direction):
    """
    Compute a quaternion that rotates [1,0,0] → direction, keeping the gate
    upright (Z-up) where possible.  Returns [w, x, y, z].
    """
    d = direction / np.linalg.norm(direction)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(world_up, d)
    right /= np.linalg.norm(right)

    up = np.cross(d, right)
    up /= np.linalg.norm(up)

    R = np.column_stack([d, right, up])  # columns = [normal, right, up]

    # Rotation matrix → quaternion (Shepperd's method)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


# ─────────────────────────────────────────────────────────────────
# Spline construction
# ─────────────────────────────────────────────────────────────────

def build_track_spline(start_pos, gate_positions):
    """
    Build a chord-length-parameterized cubic spline through
    [start_pos, gate0, gate1, ..., gateN].

    Returns:
        cs    – CubicSpline object  (call cs(t) for position, cs(t,1) for tangent)
        t_gates – parameter values at each gate (indices 1..N)
    """
    waypoints = np.vstack([start_pos.reshape(1, 3), gate_positions])
    t_knots = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        t_knots[i] = t_knots[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])

    cs = CubicSpline(t_knots, waypoints, bc_type='natural')
    return cs, t_knots[1:]   # t_knots[0] = start, 1.. = gates


# ─────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────

def draw_gate_rect(ax, pos, R, hw, color='blue', alpha=0.8):
    """Draw gate rectangle in 3D."""
    right = R[:, 1]
    up    = R[:, 2]
    corners = np.array([
        pos + hw*right + hw*up,
        pos - hw*right + hw*up,
        pos - hw*right - hw*up,
        pos + hw*right - hw*up,
        pos + hw*right + hw*up,
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            '-', color=color, linewidth=2.5, alpha=alpha)


def plot_spline_diagnostics(start_pos, gate_positions, old_rotmats,
                            spline_tangents, new_quats, gate_width, out_path):
    """
    3-panel diagnostic figure:
      Left   – 3D view with spline + old normals (red) & new normals (green)
      Middle – XY top-down
      Right  – XZ side view
    """
    cs, _ = build_track_spline(start_pos, gate_positions)
    t_fine = np.linspace(0, cs.x[-1], 500)
    pts = cs(t_fine)

    fig = plt.figure(figsize=(22, 8))

    # ── 3D ──
    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
              '--', color='#FFD700', linewidth=2, alpha=0.7, label='Spline')

    arrow = 4.0
    for i, pos in enumerate(gate_positions):
        # Old normal (red)
        old_n = old_rotmats[i][:, 0]
        ax3d.quiver(pos[0], pos[1], pos[2],
                    old_n[0]*arrow, old_n[1]*arrow, old_n[2]*arrow,
                    color='red', linewidth=1.5, arrow_length_ratio=0.12, alpha=0.6)
        # New normal (green)
        new_R = quat_to_R(*new_quats[i])
        new_n = new_R[:, 0]
        ax3d.quiver(pos[0], pos[1], pos[2],
                    new_n[0]*arrow, new_n[1]*arrow, new_n[2]*arrow,
                    color='limegreen', linewidth=2.5, arrow_length_ratio=0.15)
        # Spline tangent (cyan, thin)
        t_dir = spline_tangents[i] / np.linalg.norm(spline_tangents[i])
        ax3d.quiver(pos[0], pos[1], pos[2],
                    t_dir[0]*arrow*0.6, t_dir[1]*arrow*0.6, t_dir[2]*arrow*0.6,
                    color='cyan', linewidth=1.0, arrow_length_ratio=0.18, alpha=0.5)
        # Gate rectangle (with new rotation)
        draw_gate_rect(ax3d, pos, new_R, gate_width/2)
        ax3d.text(pos[0], pos[1], pos[2] + gate_width/2 + 0.4,
                  f"G{i}", fontsize=8, ha='center', color='blue', fontweight='bold')

    ax3d.scatter(*start_pos, color='green', s=200, marker='*',
                 zorder=10, edgecolors='black', linewidths=0.5, label='Start')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('3D — Spline Tangent Headings\n'
                   'Red=old  Green=new  Cyan=raw tangent',
                   fontsize=11, fontweight='bold')
    ax3d.legend(fontsize=8)

    # Equal aspect
    all_pts = np.vstack([gate_positions, start_pos.reshape(1, 3)])
    center = all_pts.mean(axis=0)
    rng = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3
    ax3d.set_xlim(center[0]-rng, center[0]+rng)
    ax3d.set_ylim(center[1]-rng, center[1]+rng)
    ax3d.set_zlim(max(0, center[2]-rng), center[2]+rng)
    ax3d.view_init(elev=35, azim=-60)

    # ── 2D panels ──
    for panel, (dim_a, dim_b, xlabel, ylabel, title) in enumerate([
        (0, 1, 'X', 'Y', 'Top-Down (XY)'),
        (0, 2, 'X', 'Z', 'Side View (XZ)'),
    ]):
        ax = fig.add_subplot(1, 3, panel+2)
        ax.plot(pts[:, dim_a], pts[:, dim_b],
                '--', color='#FFD700', linewidth=2, alpha=0.7, label='Spline')

        for i, pos in enumerate(gate_positions):
            old_n = old_rotmats[i][:, 0]
            new_R = quat_to_R(*new_quats[i])
            new_n = new_R[:, 0]
            t_dir = spline_tangents[i] / np.linalg.norm(spline_tangents[i])

            a_len = 3.5
            # Old normal (red)
            ax.annotate('', xy=(pos[dim_a]+old_n[dim_a]*a_len,
                                pos[dim_b]+old_n[dim_b]*a_len),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=1.5, mutation_scale=12, alpha=0.5))
            # New normal (green)
            ax.annotate('', xy=(pos[dim_a]+new_n[dim_a]*a_len,
                                pos[dim_b]+new_n[dim_b]*a_len),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='limegreen',
                                        lw=2.5, mutation_scale=15))
            # Spline tangent (cyan)
            ax.annotate('', xy=(pos[dim_a]+t_dir[dim_a]*a_len*0.6,
                                pos[dim_b]+t_dir[dim_b]*a_len*0.6),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='cyan',
                                        lw=1.0, mutation_scale=10, alpha=0.5))

            # Gate bar (perpendicular to new normal in this 2D plane)
            hw = gate_width / 2
            perp = np.array([-new_n[dim_b], new_n[dim_a]])
            if np.linalg.norm(perp) > 1e-6:
                perp /= np.linalg.norm(perp)
            p1 = np.array([pos[dim_a], pos[dim_b]]) + perp*hw
            p2 = np.array([pos[dim_a], pos[dim_b]]) - perp*hw
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color='blue', linewidth=3, alpha=0.8, zorder=4)

            # Angle between old and new
            dot_val = np.dot(old_n, new_n)
            angle = np.degrees(np.arccos(np.clip(abs(dot_val), 0, 1)))
            clr = 'lime' if angle < 10 else ('yellow' if angle < 30 else 'red')
            ax.annotate(f"G{i} Δ{angle:.0f}°", (pos[dim_a], pos[dim_b]),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=8, color=clr, fontweight='bold')

        ax.scatter(start_pos[dim_a], start_pos[dim_b],
                   color='green', s=120, marker='*', zorder=10, label='Start')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}\nRed=old  Green=new  Cyan=tangent', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle('Spline-Based Gate Heading Correction',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved diagnostic plot: {out_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fix gate rotations using spline tangent directions')
    parser.add_argument('--apply', action='store_true',
                        help='Write corrected quaternions back to the YAML config')
    args = parser.parse_args()

    config_path = os.path.expanduser(
        '~/flightmare/flightlib/configs/quadrotor_env.yaml')

    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)

    start_pos = np.array(cfg['start_pos'], dtype=np.float64)
    gates_cfg = cfg['gates']

    positions = []
    old_rotmats = []
    for g in gates_cfg:
        positions.append(np.array(g['position'], dtype=np.float64))
        q = g['rotation_quat']
        old_rotmats.append(
            quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    positions = np.array(positions)
    gate_width = float(cfg['rl']['gate_width'])

    # ── Build spline and compute tangents at each gate ──
    cs, t_gates = build_track_spline(start_pos, positions)
    spline_tangents = cs(t_gates, 1)  # first derivative at each gate

    print("=" * 70)
    print("  Spline-Based Gate Heading Analysis")
    print("=" * 70)
    print(f"  Start: {start_pos}")
    print(f"  Gates: {len(positions)}")
    print()

    new_quats = []
    for i in range(len(positions)):
        tangent = spline_tangents[i]
        tangent_dir = tangent / np.linalg.norm(tangent)

        q_new = direction_to_quat(tangent_dir)
        new_quats.append(q_new)

        R_new = quat_to_R(*q_new)
        new_normal = R_new[:, 0]
        old_normal = old_rotmats[i][:, 0]

        # Angle between old and new normals
        dot_old_new = np.dot(old_normal, new_normal)
        angle_diff = np.degrees(np.arccos(np.clip(abs(dot_old_new), 0, 1)))

        # Verify new normal matches tangent
        verify_dot = np.dot(new_normal, tangent_dir)
        verify_err = np.degrees(np.arccos(np.clip(abs(verify_dot), 0, 1)))

        status = "✓" if angle_diff < 5 else ("⚠" if angle_diff < 20 else "✗")
        print(f"  Gate {i:2d}  {status}")
        print(f"    position     : {positions[i]}")
        print(f"    spline tangent: [{tangent_dir[0]:+.5f}, {tangent_dir[1]:+.5f}, {tangent_dir[2]:+.5f}]")
        print(f"    old normal   : [{old_normal[0]:+.5f}, {old_normal[1]:+.5f}, {old_normal[2]:+.5f}]")
        print(f"    new normal   : [{new_normal[0]:+.5f}, {new_normal[1]:+.5f}, {new_normal[2]:+.5f}]")
        print(f"    Δ angle      : {angle_diff:.1f}°   (verify: {verify_err:.3f}°)")
        print(f"    new quat     : [{q_new[0]:.7f}, {q_new[1]:.7f}, {q_new[2]:.7f}, {q_new[3]:.7f}]")
        print()

    # ── Diagnostic plot ──
    plot_path = os.path.expanduser(
        '~/flightmare/flightrl/training/track_spline_heading_fix.png')
    plot_spline_diagnostics(start_pos, positions, old_rotmats,
                            spline_tangents, new_quats, gate_width, plot_path)

    # ── Optionally apply ──
    if args.apply:
        for i, q in enumerate(new_quats):
            gates_cfg[i]['rotation_quat'] = [
                float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)
        print(f"✓ Updated {config_path} with spline-based gate rotations")
    else:
        print("ℹ  Dry-run mode — no changes written.")
        print("   Re-run with --apply to write corrected quaternions to YAML.")


if __name__ == '__main__':
    main()
