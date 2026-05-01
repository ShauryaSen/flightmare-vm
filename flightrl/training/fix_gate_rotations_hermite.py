#!/usr/bin/env python3
"""
Fix gate rotations using the SAME cubic Hermite spline as the C++ reward code.

The C++ buildTrackSpline() uses cubic Hermite interpolation with tangent
directions set to gate normals scaled by chord length.  The previous
fix_gate_rotations_spline.py used scipy.CubicSpline (natural BCs) which
gives DIFFERENT tangent directions at each gate — causing a mismatch between
what the visualization shows and what the reward actually uses.

This script:
  1. Builds the exact same Hermite spline the C++ uses
  2. Computes the tangent at each gate from that Hermite spline
  3. Generates new quaternions that align each gate normal with its tangent
  4. Optionally writes them back to the YAML

Usage:
    python fix_gate_rotations_hermite.py            # dry-run (report + plot)
    python fix_gate_rotations_hermite.py --apply     # write to YAML
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
# Cubic Hermite spline (exact match with C++ buildTrackSpline)
# ─────────────────────────────────────────────────────────────────

def hermite_eval(p0, m0, p1, m1, t):
    """Evaluate cubic Hermite at t ∈ [0,1]."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1


def hermite_deriv(p0, m0, p1, m1, t):
    """Evaluate derivative of cubic Hermite at t ∈ [0,1]."""
    t2 = t * t
    dh00 = 6*t2 - 6*t
    dh10 = 3*t2 - 4*t + 1
    dh01 = -6*t2 + 6*t
    dh11 = 3*t2 - 2*t
    return dh00*p0 + dh10*m0 + dh01*p1 + dh11*m1


def build_hermite_spline(start_pos, gate_positions, gate_normals, n_samples=200):
    """
    Build the exact same cubic Hermite spline as C++ buildTrackSpline().

    Args:
        start_pos: (3,) start position
        gate_positions: (N, 3) gate positions
        gate_normals: (N, 3) gate normal directions (unit vectors)
        n_samples: samples per segment (matches kSplineSamplesPerSegment in C++)

    Returns:
        segments: list of dicts with keys 'samples', 'arc_lengths', 's_start', 's_end'
        tangent_at_gate: (N, 3) — the Hermite tangent at each gate (t=1 of its segment)
    """
    num_gates = len(gate_positions)

    # Control points: [start, gate0, gate1, ..., gateN]
    ctrl_pts = np.vstack([start_pos.reshape(1, 3), gate_positions])

    # Tangent directions: start → first gate, then gate normals
    start_dir = gate_positions[0] - start_pos
    n = np.linalg.norm(start_dir)
    if n > 1e-6:
        start_dir /= n
    tangent_dirs = np.vstack([start_dir.reshape(1, 3), gate_normals])

    segments = []
    tangent_at_gate = []
    global_arc = 0.0

    for seg_i in range(num_gates):
        p0 = ctrl_pts[seg_i]
        p1 = ctrl_pts[seg_i + 1]
        chord = np.linalg.norm(p1 - p0)

        # Scale tangents by chord length (exact C++ match)
        m0 = tangent_dirs[seg_i] * chord
        m1 = tangent_dirs[seg_i + 1] * chord

        samples = np.zeros((n_samples, 3))
        arc_lengths = np.zeros(n_samples)

        s_start = global_arc
        samples[0] = p0
        arc_lengths[0] = global_arc

        for j in range(1, n_samples):
            t = j / (n_samples - 1)
            samples[j] = hermite_eval(p0, m0, p1, m1, t)
            global_arc += np.linalg.norm(samples[j] - samples[j-1])
            arc_lengths[j] = global_arc

        s_end = global_arc

        segments.append({
            'samples': samples,
            'arc_lengths': arc_lengths,
            's_start': s_start,
            's_end': s_end,
            'p0': p0, 'p1': p1, 'm0': m0, 'm1': m1,
        })

        # Tangent at t=1 (the gate end of this segment)
        tang = hermite_deriv(p0, m0, p1, m1, 1.0)
        tangent_at_gate.append(tang / np.linalg.norm(tang))

    return segments, np.array(tangent_at_gate)


# ─────────────────────────────────────────────────────────────────
# Iterative convergence: re-derive gate normals from Hermite tangents
# ─────────────────────────────────────────────────────────────────

def converge_gate_normals(start_pos, gate_positions, initial_normals,
                          max_iters=20, tol=1e-6):
    """
    Iteratively converge gate normals to match Hermite spline tangents.

    The Hermite spline uses gate normals as tangent directions, so changing
    the normals changes the spline, which changes the tangents.  We iterate
    until the tangent directions stop changing.

    Returns:
        final_normals: (N, 3) converged normal directions
        final_quats: list of [w, x, y, z] quaternions
        segments: final spline segments (for plotting)
    """
    normals = initial_normals.copy()
    num_gates = len(gate_positions)

    for iteration in range(max_iters):
        segments, tangents = build_hermite_spline(
            start_pos, gate_positions, normals)

        # Check convergence
        max_delta = 0.0
        for i in range(num_gates):
            delta = np.arccos(np.clip(np.dot(normals[i], tangents[i]), -1, 1))
            max_delta = max(max_delta, delta)

        if max_delta < tol:
            print(f"  Converged in {iteration+1} iterations "
                  f"(max_delta={np.degrees(max_delta):.4f}°)")
            break

        normals = tangents.copy()
    else:
        print(f"  WARNING: Did not converge after {max_iters} iters "
              f"(max_delta={np.degrees(max_delta):.4f}°)")

    # Compute final quaternions
    final_quats = []
    for i in range(num_gates):
        q = direction_to_quat(normals[i])
        final_quats.append(q)

    return normals, final_quats, segments


# ─────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────

def plot_diagnostics(start_pos, gate_positions, old_normals, new_normals,
                     segments, new_quats, gate_width, out_path):
    """3-panel diagnostic: 3D, XY top-down, XZ side."""
    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor('#0a0a1a')

    # Collect all spline points
    all_spline_pts = np.vstack([seg['samples'] for seg in segments])

    # ── 3D ──
    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.set_facecolor('#0a0a1a')

    # Spline (gold)
    ax3d.plot(all_spline_pts[:, 0], all_spline_pts[:, 1], all_spline_pts[:, 2],
              '-', color='#FFD700', linewidth=2.5, alpha=0.85, label='Hermite Spline')

    arrow = 4.0
    for i, pos in enumerate(gate_positions):
        # Old normal (red, faded)
        ax3d.quiver(pos[0], pos[1], pos[2],
                    old_normals[i, 0]*arrow, old_normals[i, 1]*arrow, old_normals[i, 2]*arrow,
                    color='red', linewidth=1.5, arrow_length_ratio=0.12, alpha=0.4)
        # New normal (green)
        ax3d.quiver(pos[0], pos[1], pos[2],
                    new_normals[i, 0]*arrow, new_normals[i, 1]*arrow, new_normals[i, 2]*arrow,
                    color='limegreen', linewidth=2.5, arrow_length_ratio=0.15)

        # Gate rectangle
        R_new = quat_to_R(*new_quats[i])
        right = R_new[:, 1]
        up = R_new[:, 2]
        hw = gate_width / 2
        corners = np.array([
            pos + hw*right + hw*up,
            pos - hw*right + hw*up,
            pos - hw*right - hw*up,
            pos + hw*right - hw*up,
            pos + hw*right + hw*up,
        ])
        ax3d.plot(corners[:, 0], corners[:, 1], corners[:, 2],
                  '-', color='#4488ff', linewidth=2.5, alpha=0.8)

        ax3d.text(pos[0], pos[1], pos[2] + hw + 0.4,
                  f"G{i}", fontsize=8, ha='center', color='#4488ff', fontweight='bold')

    ax3d.scatter(*start_pos, color='#FFD700', s=200, marker='*',
                 zorder=10, edgecolors='black', linewidths=0.5, label='Start')

    ax3d.set_xlabel('X', color='#999'); ax3d.set_ylabel('Y', color='#999')
    ax3d.set_zlabel('Z', color='#999')
    ax3d.set_title('3D — Hermite Spline Headings\nRed=old  Green=new (converged)',
                   fontsize=11, fontweight='bold', color='#ddd')
    ax3d.legend(fontsize=8)

    all_pts = np.vstack([gate_positions, start_pos.reshape(1, 3)])
    center = all_pts.mean(axis=0)
    rng = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.3
    ax3d.set_xlim(center[0]-rng, center[0]+rng)
    ax3d.set_ylim(center[1]-rng, center[1]+rng)
    ax3d.set_zlim(max(0, center[2]-rng), center[2]+rng)
    ax3d.view_init(elev=35, azim=-60)
    ax3d.tick_params(colors='#666')

    # ── 2D panels ──
    for panel, (dim_a, dim_b, xlabel, ylabel, title) in enumerate([
        (0, 1, 'X', 'Y', 'Top-Down (XY)'),
        (0, 2, 'X', 'Z', 'Side View (XZ)'),
    ]):
        ax = fig.add_subplot(1, 3, panel+2)
        ax.set_facecolor('#0a0a1a')

        # Spline (gold)
        ax.plot(all_spline_pts[:, dim_a], all_spline_pts[:, dim_b],
                '-', color='#FFD700', linewidth=2.5, alpha=0.85, label='Hermite Spline')

        for i, pos in enumerate(gate_positions):
            a_len = 3.5
            # Old normal (red)
            ax.annotate('', xy=(pos[dim_a]+old_normals[i, dim_a]*a_len,
                                pos[dim_b]+old_normals[i, dim_b]*a_len),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=1.5, mutation_scale=12, alpha=0.4))
            # New normal (green)
            ax.annotate('', xy=(pos[dim_a]+new_normals[i, dim_a]*a_len,
                                pos[dim_b]+new_normals[i, dim_b]*a_len),
                        xytext=(pos[dim_a], pos[dim_b]),
                        arrowprops=dict(arrowstyle='->', color='limegreen',
                                        lw=2.5, mutation_scale=15))

            # Gate bar perpendicular to new normal
            R_new = quat_to_R(*new_quats[i])
            new_n = R_new[:, 0]
            hw = gate_width / 2
            perp = np.array([-new_n[dim_b], new_n[dim_a]])
            if np.linalg.norm(perp) > 1e-6:
                perp /= np.linalg.norm(perp)
            p1 = np.array([pos[dim_a], pos[dim_b]]) + perp*hw
            p2 = np.array([pos[dim_a], pos[dim_b]]) - perp*hw
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color='#4488ff', linewidth=3, alpha=0.8, zorder=4)

            # Angle between old and new
            dot_val = np.dot(old_normals[i], new_normals[i])
            angle = np.degrees(np.arccos(np.clip(abs(dot_val), 0, 1)))
            clr = '#44ff44' if angle < 10 else ('#ffaa00' if angle < 30 else '#ff4444')
            ax.annotate(f"G{i} Δ{angle:.0f}°", (pos[dim_a], pos[dim_b]),
                        textcoords='offset points', xytext=(8, 8),
                        fontsize=8, color=clr, fontweight='bold')

        ax.scatter(start_pos[dim_a], start_pos[dim_b],
                   color='#FFD700', s=120, marker='*', zorder=10, label='Start')
        ax.set_xlabel(xlabel, fontsize=11, color='#999')
        ax.set_ylabel(ylabel, fontsize=11, color='#999')
        ax.set_title(f'{title}\nRed=old  Green=new (converged)', fontsize=10,
                     fontweight='bold', color='#ddd')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='gray')
        ax.legend(fontsize=7)
        ax.tick_params(colors='#666')
        for spine in ax.spines.values():
            spine.set_color('#333')

    fig.suptitle('Hermite Spline Gate Heading Correction (matches C++ reward)',
                 fontsize=14, fontweight='bold', y=1.02, color='white')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Saved diagnostic plot: {out_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fix gate rotations using Hermite spline tangents (matches C++)')
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
    old_normals = []
    for g in gates_cfg:
        positions.append(np.array(g['position'], dtype=np.float64))
        q = g['rotation_quat']
        R = quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        old_normals.append(R[:, 0] / np.linalg.norm(R[:, 0]))
    positions = np.array(positions)
    old_normals = np.array(old_normals)
    gate_width = float(cfg['rl']['gate_width'])

    print("=" * 70)
    print("  Hermite Spline Gate Heading Correction")
    print("  (Matches C++ buildTrackSpline exactly)")
    print("=" * 70)
    print(f"  Start: {start_pos}")
    print(f"  Gates: {len(positions)}")
    print()

    # Converge: iteratively update normals until spline tangents == normals
    new_normals, new_quats, segments = converge_gate_normals(
        start_pos, positions, old_normals)

    # Print report
    for i in range(len(positions)):
        angle_diff = np.degrees(np.arccos(
            np.clip(abs(np.dot(old_normals[i], new_normals[i])), 0, 1)))
        status = "✓" if angle_diff < 5 else ("⚠" if angle_diff < 20 else "✗")
        print(f"  Gate {i:2d}  {status}")
        print(f"    position     : {positions[i]}")
        print(f"    old normal   : [{old_normals[i, 0]:+.5f}, {old_normals[i, 1]:+.5f}, {old_normals[i, 2]:+.5f}]")
        print(f"    new normal   : [{new_normals[i, 0]:+.5f}, {new_normals[i, 1]:+.5f}, {new_normals[i, 2]:+.5f}]")
        print(f"    Δ angle      : {angle_diff:.1f}°")
        print(f"    new quat     : [{new_quats[i][0]:.10f}, {new_quats[i][1]:.10f}, {new_quats[i][2]:.10f}, {new_quats[i][3]:.10f}]")
        print()

    # Total spline arc length
    total_arc = segments[-1]['s_end']
    print(f"  Total spline arc length: {total_arc:.1f} m")

    # Diagnostic plot
    plot_path = os.path.expanduser(
        '~/flightmare/flightrl/training/track_hermite_heading_fix.png')
    plot_diagnostics(start_pos, positions, old_normals, new_normals,
                     segments, new_quats, gate_width, plot_path)

    # Optionally apply
    if args.apply:
        for i, q in enumerate(new_quats):
            gates_cfg[i]['rotation_quat'] = [
                float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)
        print(f"✓ Updated {config_path} with Hermite-converged gate rotations")
    else:
        print("ℹ  Dry-run mode — no changes written.")
        print("   Re-run with --apply to write corrected quaternions to YAML.")


if __name__ == '__main__':
    main()
