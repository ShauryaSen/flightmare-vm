#!/usr/bin/env python3
"""
Gate Geometry Diagnostic — Visual and numerical verification of gate normals.

Generates:
  1. Top-down (XY) plot with gate normals + segment paths + approach arrows
  2. Side view (XZ) 
  3. 3D perspective
  4. Numerical analysis table

Tests BOTH quaternion sets (AlphaPilot original vs current auto-face).

Usage: python gate_geometry_check.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

# ── Gate Data ──────────────────────────────────────────────────

gate_positions = np.array([
    [17.25, 3.63, 6.31],    # G1
    [16.75, 39.57, 6.76],   # G2
    [2.33, 27.87, 2.55],    # G3
    [2.20, 9.00, 1.99],     # G4
    [-7.31, -12.14, 3.23],  # G5
    [-7.61, -28.47, 2.64],  # G6
    [-0.01, -33.91, 2.10],  # G7
    [5.87, -28.44, 2.57],   # G8
    [7.25, -11.79, 2.55],   # G9
    [-10.23, 7.77, 2.02],   # G10
    [-10.05, 30.62, 2.90],  # G11
])

start_pos = np.array([17.5, -15.0, 6.31])

# Current YAML quaternions (auto-face path from track_visualizer)
current_quats = [
    [0.9771817, -0.01343000920589536, 0.065047637429392, 0.2017531],
    [0.2280067, 0.11531715801674725, -0.027206581810986515, 0.9664237],
    [0.5335431, -0.05887161003653859, -0.03726493386451581, -0.8428981],
    [0.6272654, 0.004774282673377631, 0.003845420905743276, -0.7787814],
    [0.6250068, 0.0034548347958006805, 0.002766175553015647, -0.7806067],
    [0.8507814, -0.013872470162431589, -0.022487043651383685, -0.5248551],
    [0.9994835, -0.0000012978660909514265, 0.000040364775467993254, 0.0321368],
    [0.8473638, -0.008039958766361295, 0.012834988435292868, 0.5307967],
    [0.5841497, 0.0047098038495232515, -0.0033897797781538854, 0.8116252],
    [0.5687835, -0.004041671728656527, 0.0027950308781103544, 0.8224726],
    [0.9773087, -0.015891114827876775, 0.0793354135254295, 0.1957578],
]

# AlphaPilot original quaternions
alphapilot_quats = [
    [0.7071270, 0.0, 0.0, -0.7070865],
    [0.9238791, 0.0, 0.0, -0.3826845],
    [0.7071068, 0.0, 0.0, 0.7071068],
    [0.7071068, 0.0, 0.0, 0.7071068],
    [0.7071068, 0.0, 0.0, 0.7071068],
    [-0.3826836, 0.0, 0.0, -0.9238794],
    [-0.0000263, 0.0, 0.0, -1.0000000],
    [0.3826835, 0.0, 0.0, -0.9238795],
    [0.7071068, 0.0, 0.0, -0.7071068],
    [0.7071270, 0.0, 0.0, -0.7070865],
    [0.7071323, 0.0, 0.0, -0.7070812],
]

num_gates = len(gate_positions)


def quat_to_R(w, x, y, z):
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def compute_normals(quats):
    """Compute gate normals as R * [1,0,0] (C++ convention)."""
    normals = []
    for q in quats:
        R = quat_to_R(*q)
        normal = R[:, 0]  # First column = R * [1,0,0]
        normal = normal / np.linalg.norm(normal)
        normals.append(normal)
    return np.array(normals)


def compute_approach_dirs():
    """Compute approach direction for each gate (from previous gate)."""
    dirs = []
    for i in range(num_gates):
        prev = gate_positions[(i - 1) % num_gates]
        curr = gate_positions[i]
        seg = curr - prev
        seg_dir = seg / np.linalg.norm(seg)
        dirs.append(seg_dir)
    return np.array(dirs)


def plot_track(quats, quat_name, ax, view='topdown'):
    """Plot gates, normals, and segments on a given axes."""
    normals = compute_normals(quats)
    approach_dirs = compute_approach_dirs()

    if view == 'topdown':
        dim_a, dim_b = 0, 1
        xlabel, ylabel = 'X (m)', 'Y (m)'
    elif view == 'side':
        dim_a, dim_b = 0, 2
        xlabel, ylabel = 'X (m)', 'Z (m)'
    else:
        dim_a, dim_b = 1, 2
        xlabel, ylabel = 'Y (m)', 'Z (m)'

    # Draw segment path (green dashed)
    for i in range(num_gates):
        next_i = (i + 1) % num_gates
        ax.plot([gate_positions[i, dim_a], gate_positions[next_i, dim_a]],
                [gate_positions[i, dim_b], gate_positions[next_i, dim_b]],
                'g--', alpha=0.3, linewidth=1)

    # Draw start → G1 segment
    ax.plot([start_pos[dim_a], gate_positions[0, dim_a]],
            [start_pos[dim_b], gate_positions[0, dim_b]],
            'y--', alpha=0.4, linewidth=1)

    # Draw gates with normals
    arrow_len = 3.0
    for i in range(num_gates):
        pos = gate_positions[i]
        normal = normals[i]
        approach = approach_dirs[i]
        dot_val = np.dot(normal, approach)

        # Color by alignment
        if dot_val > 0.5:
            color = '#44ff44'
        elif dot_val > 0:
            color = '#aaff44'
        elif dot_val > -0.5:
            color = '#ffaa00'
        else:
            color = '#ff4444'

        # Gate marker
        ax.plot(pos[dim_a], pos[dim_b], 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        # Normal arrow (red = fly-through direction from C++)
        ax.annotate('', xy=(pos[dim_a] + normal[dim_a]*arrow_len,
                            pos[dim_b] + normal[dim_b]*arrow_len),
                    xytext=(pos[dim_a], pos[dim_b]),
                    arrowprops=dict(arrowstyle='->', color='#ff4444',
                                   lw=2.0, mutation_scale=15),
                    zorder=4)

        # Approach arrow (magenta = direction drone arrives from)
        ax.annotate('', xy=(pos[dim_a], pos[dim_b]),
                    xytext=(pos[dim_a] - approach[dim_a]*arrow_len*0.8,
                            pos[dim_b] - approach[dim_b]*arrow_len*0.8),
                    arrowprops=dict(arrowstyle='->', color='#ff44ff',
                                   lw=1.5, mutation_scale=12, alpha=0.6),
                    zorder=3)

        # Gate label with dot product
        ax.text(pos[dim_a] + 0.8, pos[dim_b] + 0.8,
                f'G{i+1}\n{dot_val:+.2f}',
                fontsize=7, fontweight='bold', color=color,
                ha='left', va='bottom', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Start position
    ax.plot(start_pos[dim_a], start_pos[dim_b], '*', color='#ffaa00',
            markersize=15, markeredgecolor='white', markeredgewidth=0.5, zorder=5)
    ax.text(start_pos[dim_a] + 1, start_pos[dim_b] + 1, 'START',
            fontsize=8, fontweight='bold', color='#ffaa00')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{quat_name}', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='gray')
    ax.set_facecolor('#0a0a1a')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.tick_params(colors='#999')
    ax.xaxis.label.set_color('#999')
    ax.yaxis.label.set_color('#999')
    ax.title.set_color('#ddd')


def print_analysis(quats, quat_name):
    """Print numerical analysis of gate normals vs approach directions."""
    normals = compute_normals(quats)
    approach_dirs = compute_approach_dirs()

    print(f"\n{'='*80}")
    print(f"  {quat_name}")
    print(f"{'='*80}")
    print(f"  {'Gate':<5} {'Normal (R*[1,0,0])':<30} {'Approach dir':<30} {'Dot':<8} {'Angle':<8}")
    print(f"  {'-'*75}")

    all_dots = []
    for i in range(num_gates):
        n = normals[i]
        a = approach_dirs[i]
        d = np.dot(n, a)
        angle = np.degrees(np.arccos(np.clip(d, -1, 1)))
        all_dots.append(d)
        status = '✓' if d > 0.3 else ('~' if d > 0 else '✗')
        print(f"  G{i+1:<4} ({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})      "
              f"({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})      "
              f"{d:+.3f}  {angle:5.1f}°  {status}")

    all_dots = np.array(all_dots)
    print(f"\n  Summary:")
    print(f"    Mean dot product:     {all_dots.mean():+.3f}")
    print(f"    All positive:         {np.all(all_dots > 0)}")
    print(f"    Positive gates:       {np.sum(all_dots > 0)}/{num_gates}")
    print(f"    Min dot:              {all_dots.min():+.3f} (G{np.argmin(all_dots)+1})")
    print(f"    Max dot:              {all_dots.max():+.3f} (G{np.argmax(all_dots)+1})")

    if np.all(all_dots > 0):
        print(f"\n  ✅ All gate normals align with approach directions!")
        print(f"     The convention normal = R*[1,0,0] is correct for these quaternions.")
    elif np.all(all_dots < 0):
        print(f"\n  ⚠️ All normals are REVERSED vs approach. May need R*[-1,0,0] convention.")
    else:
        print(f"\n  ❌ Mixed alignment — some normals correct, some backwards.")
        print(f"     This likely means the quaternions or axis convention is inconsistent.")

    return all_dots


def main():
    print("="*80)
    print("  GATE GEOMETRY DIAGNOSTIC")
    print("  Verifying that gate normals point in the correct fly-through direction")
    print("="*80)
    print(f"\n  Track: {num_gates} gates, start at ({start_pos[0]}, {start_pos[1]}, {start_pos[2]})")
    print(f"  C++ convention: gate_normal = R_gate * [1, 0, 0]")

    # ── Numerical analysis ──
    dots_current = print_analysis(current_quats, "Current YAML (auto-face path)")
    dots_alpha = print_analysis(alphapilot_quats, "AlphaPilot original")

    # ── Generate comparison plot ──
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.patch.set_facecolor('#0a0a1a')

    plot_track(current_quats, 'Current YAML — Top Down (XY)', axes[0, 0], view='topdown')
    plot_track(alphapilot_quats, 'AlphaPilot Original — Top Down (XY)', axes[0, 1], view='topdown')
    plot_track(current_quats, 'Current YAML — Side View (XZ)', axes[1, 0], view='side')
    plot_track(alphapilot_quats, 'AlphaPilot Original — Side View (XZ)', axes[1, 1], view='side')

    # Legend
    fig.text(0.5, 0.01,
             'Red arrows = gate normal (R×[1,0,0]) = fly-through dir  |  '
             'Magenta arrows = incoming approach  |  '
             'Green dashed = segment path  |  '
             'Number = dot(normal, approach)',
             ha='center', fontsize=10, color='#aaa',
             bbox=dict(boxstyle='round', facecolor='#111', alpha=0.8))

    fig.suptitle('Gate Normal Alignment Diagnostic\n'
                 'Green labels = correct alignment (dot > 0)  |  '
                 'Red labels = reversed (dot < 0)',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), 'gate_geometry_diagnostic.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✓ Saved diagnostic plot: {out_path}")

    # ── Verdict ──
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")
    if np.all(dots_current > 0):
        print(f"  ✅ Current YAML quaternions: ALL normals align with approach directions.")
        print(f"     These quaternions are CORRECT for the R*[1,0,0] convention.")
    if not np.all(dots_alpha > 0):
        print(f"  ⚠️  AlphaPilot quaternions: NOT all normals align with R*[1,0,0].")
        print(f"     AlphaPilot likely uses a DIFFERENT axis convention (e.g., R*[0,1,0] or R*[-1,0,0]).")
    print(f"\n  If the current quats are correct but training still fails,")
    print(f"  the problem is NOT the gate orientations. Look at:")
    print(f"    1. Segment progress computation")
    print(f"    2. Gate pass detection logic")
    print(f"    3. Reward scaling / magnitudes")
    print(f"    4. Observation space correctness")
    print(f"    5. Training hyperparameters")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
