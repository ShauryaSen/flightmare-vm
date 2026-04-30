#!/usr/bin/env python3
"""
Gate Collision Diagnostic — Shows EXACTLY why the drone always terminates.

Computes the geometry of gate-pass detection for every gate and shows:
1. Where the "trigger plane" is vs where the gate physically is
2. What d_p (lateral distance) the drone would see at the trigger point
3. Whether gate_width/2 = 1.5m is sufficient opening radius

The key insight:
  - Gate-pass TRIGGER: s(pos) >= seg_len  (projection on SEGMENT direction)
  - Gate-pass CHECK:   d_p < gate_width/2  (distance from GATE NORMAL axis)
  - These use DIFFERENT reference frames when gate_normal ≠ segment_direction

Usage: python gate_collision_diagnostic.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os


# ── Gate Data (from quadrotor_env.yaml) ──

gate_positions = np.array([
    [17.25, 3.63, 6.31],    # G0
    [16.75, 39.57, 6.76],   # G1
    [2.33, 27.87, 2.55],    # G2
    [2.20, 9.00, 1.99],     # G3
    [-7.31, -12.14, 3.23],  # G4
    [-7.61, -28.47, 2.64],  # G5
    [-0.01, -33.91, 2.10],  # G6
    [5.87, -28.44, 2.57],   # G7
    [7.25, -11.79, 2.55],   # G8
    [-10.23, 7.77, 2.02],   # G9
    [-10.05, 30.62, 2.90],  # G10
])

start_pos = np.array([17.5, -15.0, 6.31])

quats = [
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

num_gates = len(gate_positions)
gate_width = 3.0  # from YAML


def quat_to_R(w, x, y, z):
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def get_gate_normal(gate_idx):
    R = quat_to_R(*quats[gate_idx])
    normal = R[:, 0]  # R * [1,0,0]
    return normal / np.linalg.norm(normal)


def get_segment(gate_idx):
    """Get the segment from previous gate to this gate (mirrors C++ logic)."""
    if gate_idx == 0:
        g_prev = start_pos
    else:
        g_prev = gate_positions[(gate_idx - 1 + num_gates) % num_gates]
    g_curr = gate_positions[gate_idx]
    return g_prev, g_curr


def segment_progress(pos, gate_idx):
    """Mirror of C++ segment_progress function."""
    g1, g2 = get_segment(gate_idx)
    seg = g2 - g1
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        return 0.0
    raw = np.dot(pos - g1, seg) / seg_len
    return max(0.0, min(raw, seg_len + 0.5))


def compute_d_p(pos, gate_idx):
    """Mirror of C++ d_p computation — distance from gate normal axis."""
    g_curr = gate_positions[gate_idx]
    n_curr = get_gate_normal(gate_idx)
    d_n = np.dot(pos - g_curr, n_curr)  # signed dist along normal
    d_p = np.linalg.norm(pos - g_curr - d_n * n_curr)  # perp dist from axis
    return d_p, d_n


def analyze_gate(gate_idx):
    """Full geometric analysis of gate-pass detection for one gate."""
    g_prev, g_curr = get_segment(gate_idx)
    seg = g_curr - g_prev
    seg_len = np.linalg.norm(seg)
    seg_dir = seg / seg_len
    
    normal = get_gate_normal(gate_idx)
    
    # The segment-based "trigger plane" is at s(pos) = seg_len
    # This plane is perpendicular to seg_dir and passes through g_curr
    # (since projecting g_curr onto the segment gives exactly seg_len)
    
    # The gate-normal plane passes through g_curr perpendicular to normal
    
    # Angle between segment direction and gate normal
    dot_sn = np.dot(seg_dir, normal)
    angle_sn = np.degrees(np.arccos(np.clip(abs(dot_sn), 0, 1)))
    
    # When the drone is exactly at g_curr (gate center):
    # s(g_curr) = dot(g_curr - g_prev, seg) / seg_len = seg_len ✓ (triggers)
    # d_p(g_curr) = |g_curr - g_curr - 0 * n| = 0 ✓ (passes)
    # So if the drone hits the gate CENTER exactly, it works!
    
    # But what if the drone arrives at the trigger plane (s=seg_len) 
    # but not at gate center? Let's check a point on the trigger plane
    # that's offset laterally...
    
    # Point on trigger plane, offset by 1m perpendicular to seg_dir:
    # We need a vector perpendicular to seg_dir
    perp = np.cross(seg_dir, [0, 0, 1])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(seg_dir, [0, 1, 0])
    perp = perp / np.linalg.norm(perp)
    
    # Point at gate center (on trigger plane) + 1m lateral offset
    test_point = g_curr + 1.0 * perp
    s_test = segment_progress(test_point, gate_idx)
    d_p_test, d_n_test = compute_d_p(test_point, gate_idx)
    
    # If the drone is exactly on the segment line but 1m past the gate:
    overshoot_point = g_curr + 0.3 * seg_dir
    s_over = segment_progress(overshoot_point, gate_idx)
    d_p_over, d_n_over = compute_d_p(overshoot_point, gate_idx)
    
    return {
        'gate_idx': gate_idx,
        'g_prev': g_prev,
        'g_curr': g_curr,
        'seg_dir': seg_dir,
        'seg_len': seg_len,
        'normal': normal,
        'dot_seg_normal': dot_sn,
        'angle_seg_normal_deg': angle_sn,
        # Test: laterally offset on trigger plane
        'test_lateral_s': s_test,
        'test_lateral_dp': d_p_test,
        'test_lateral_dn': d_n_test,
        # Test: overshoot along segment
        'test_overshoot_s': s_over,
        'test_overshoot_dp': d_p_over,
        'test_overshoot_dn': d_n_over,
    }


def main():
    print("=" * 90)
    print("  GATE COLLISION DIAGNOSTIC")
    print("  Analyzing why gate-pass detection always triggers collision")
    print("=" * 90)
    
    print(f"\n  Config: gate_width = {gate_width}m, gate_width/2 = {gate_width/2}m (pass threshold)")
    print(f"  Trigger: s(pos) >= seg_len  (projection on segment)")
    print(f"  Check:   d_p < gate_width/2 (distance from gate NORMAL axis)")
    print()
    
    print(f"  {'Gate':>4} {'SegLen':>7} {'Seg→Norm':>9} {'Angle':>7} "
          f"{'d_p@center':>10} {'d_p@1m_lat':>11} {'d_p@overshoot':>14} "
          f"{'Pass?':>6}")
    print(f"  {'-'*85}")
    
    results = []
    for i in range(num_gates):
        r = analyze_gate(i)
        results.append(r)
        
        # d_p at gate center should be 0
        d_p_center, _ = compute_d_p(r['g_curr'], i)
        
        # Would a drone at gate center pass?
        pass_center = "✓" if d_p_center < gate_width / 2 else "✗"
        # Would a drone 1m laterally offset pass?
        pass_lateral = "✓" if r['test_lateral_dp'] < gate_width / 2 else "✗"
        
        print(f"  G{i:<3} {r['seg_len']:7.1f} {r['dot_seg_normal']:+9.4f} "
              f"{r['angle_seg_normal_deg']:6.1f}° "
              f"{d_p_center:10.3f}  {r['test_lateral_dp']:10.3f}({pass_lateral})  "
              f"{r['test_overshoot_dp']:10.3f}     {pass_center}")
    
    # ── KEY ANALYSIS: What happens at the trigger boundary ──
    print(f"\n{'='*90}")
    print(f"  CRITICAL ANALYSIS: Frame Mismatch")
    print(f"{'='*90}")
    
    print(f"""
  The gate-pass detection has TWO independent checks:
  
  1. TRIGGER: s(pos) >= seg_len
     - s(pos) = dot(pos - g_prev, seg_dir) 
     - This measures progress along the SEGMENT LINE (g_prev → g_curr)
     - Triggers when the drone's projection reaches g_curr
  
  2. CHECK: d_p < gate_width/2
     - d_p = |pos - g_curr - d_n * n_curr|
     - d_n = dot(pos - g_curr, n_curr) — distance along GATE NORMAL
     - d_p = perpendicular distance from the GATE NORMAL AXIS through g_curr
     
  PROBLEM: When seg_dir ≠ normal, the "perpendicular plane" for the 
  trigger (⊥ to seg_dir) differs from the gate plane (⊥ to normal).
  
  At the EXACT MOMENT the trigger fires (s = seg_len), the drone is on
  a plane perpendicular to seg_dir. Its d_p is measured perpendicular
  to normal. If these axes are misaligned, d_p gives a WRONG reading.
  """)
    
    # Simulate a drone flying straight along the segment
    print(f"\n  SIMULATION: Drone flying along segment direction")
    print(f"  {'Gate':>4} {'At trigger → d_p':>17} {'d_n':>8} {'gate_w/2':>8} {'RESULT':>12}")
    print(f"  {'-'*60}")
    
    bad_gates = []
    for i in range(num_gates):
        r = results[i]
        # Drone at g_curr (which is exactly where s = seg_len)
        d_p_at_trigger, d_n_at_trigger = compute_d_p(r['g_curr'], i)
        
        result = "PASS ✓" if d_p_at_trigger < gate_width / 2 else "COLLISION ✗"
        if d_p_at_trigger >= gate_width / 2:
            bad_gates.append(i)
        
        print(f"  G{i:<3} {d_p_at_trigger:17.4f} {d_n_at_trigger:8.4f} "
              f"{gate_width/2:8.2f}  {result}")
    
    # Now simulate a drone approaching slightly off-axis
    print(f"\n  SIMULATION: Drone arriving 0.5m off the segment line")
    print(f"  {'Gate':>4} {'d_p (at trigger)':>17} {'RESULT':>12}")
    print(f"  {'-'*40}")
    
    for i in range(num_gates):
        r = results[i]
        seg_dir = r['seg_dir']
        perp = np.cross(seg_dir, [0, 0, 1])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(seg_dir, [0, 1, 0])
        perp = perp / np.linalg.norm(perp)
        
        # Drone at gate center + 0.5m lateral offset
        test_pos = r['g_curr'] + 0.5 * perp
        s_test = segment_progress(test_pos, i)
        d_p_test, _ = compute_d_p(test_pos, i)
        
        # Check: does s(test_pos) >= seg_len?
        triggers = s_test >= r['seg_len']
        passes = d_p_test < gate_width / 2
        
        if triggers:
            result = "PASS ✓" if passes else "COLLISION ✗"
        else:
            result = "no trigger"
        
        print(f"  G{i:<3} {d_p_test:17.4f}  {result}")
    
    # ── The REAL problem: check what d_p looks like for a drone
    # flying THROUGH the gate opening on the normal axis ──
    print(f"\n{'='*90}")
    print(f"  REAL SCENARIO: Drone flies through gate on the NORMAL axis")
    print(f"  (approaching from normal direction, passing through center)")
    print(f"{'='*90}")
    
    print(f"\n  {'Gate':>4} {'s at crossing':>13} {'seg_len':>8} {'Triggers?':>10} "
          f"{'d_p':>8} {'Passes?':>8}")
    print(f"  {'-'*65}")
    
    for i in range(num_gates):
        r = results[i]
        # Drone at gate center — perfect pass through
        s_at_center = segment_progress(r['g_curr'], i)
        triggers = s_at_center >= r['seg_len']
        d_p, _ = compute_d_p(r['g_curr'], i)
        passes = d_p < gate_width / 2
        
        trigger_str = "YES ✓" if triggers else "NO ✗"
        pass_str = "YES ✓" if passes else "NO ✗"
        
        # Now check 0.1m before center on normal axis
        # (drone just barely hasn't crossed yet)
        pos_before = r['g_curr'] - 0.1 * r['normal']
        s_before = segment_progress(pos_before, i)
        
        # And 0.1m after center on normal axis
        pos_after = r['g_curr'] + 0.1 * r['normal']
        s_after = segment_progress(pos_after, i)
        
        # Was there a transition?
        transition = s_before < r['seg_len'] and s_after >= r['seg_len']
        
        print(f"  G{i:<3} {s_at_center:13.3f} {r['seg_len']:8.3f}  {trigger_str:>10} "
              f"{d_p:8.4f}  {pass_str:>8}  "
              f"{'(transition ✓)' if transition else '(NO transition along normal ✗)'}")
    
    if bad_gates:
        print(f"\n  ⚠️  Gates with issues: {bad_gates}")
    
    print(f"\n{'='*90}")
    print(f"  CONCLUSION")
    print(f"{'='*90}")
    print(f"""
  The gate collision issue depends on whether the segment direction and 
  gate normal are aligned. When they're closely aligned, the system works.
  When they diverge, the trigger plane and gate plane become different 
  surfaces, and the d_p check is evaluated at the wrong location.
  
  The drone can fly straight through the gate opening on the normal axis,
  but if its projection on the segment hasn't reached seg_len, the trigger
  never fires (gate pass is missed). Conversely, if the drone passes the 
  trigger plane at a different position than the gate plane, d_p gives a 
  wrong reading.
    """)
    
    # ── Generate visualization ──
    generate_diagnostic_plot(results)


def generate_diagnostic_plot(results):
    """Generate a detailed visualization showing gate normals, segment 
    directions, trigger planes, and pass/collision zones."""
    
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    fig.patch.set_facecolor('#0a0a1a')
    
    for ax_idx, (ax, dim_a, dim_b, xlabel, ylabel, title) in enumerate([
        (axes[0], 0, 1, 'X (m)', 'Y (m)', 'Top-Down View (XY)'),
        (axes[1], 0, 2, 'X (m)', 'Z (m)', 'Side View (XZ)'),
    ]):
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.tick_params(colors='#999')
        ax.xaxis.label.set_color('#bbb')
        ax.yaxis.label.set_color('#bbb')
        ax.title.set_color('#eee')
        
        # Draw segment paths (white dashed)
        for i in range(num_gates):
            r = results[i]
            ax.plot([r['g_prev'][dim_a], r['g_curr'][dim_a]],
                    [r['g_prev'][dim_b], r['g_curr'][dim_b]],
                    '--', color='#555', linewidth=1, alpha=0.6)
        
        # Draw gates with normals and segment directions
        arrow_len = 4.0
        for i in range(num_gates):
            r = results[i]
            pos = r['g_curr']
            normal = r['normal']
            seg_dir = r['seg_dir']
            
            dot = r['dot_seg_normal']
            angle = r['angle_seg_normal_deg']
            
            # Color gate by alignment quality
            if angle < 10:
                gate_color = '#00ff88'
                status = 'OK'
            elif angle < 30:
                gate_color = '#ffcc00'
                status = 'WARN'
            else:
                gate_color = '#ff4444'
                status = 'BAD'
            
            # Gate center marker
            ax.plot(pos[dim_a], pos[dim_b], 'o', color=gate_color,
                    markersize=12, markeredgecolor='white',
                    markeredgewidth=1, zorder=10)
            
            # Gate NORMAL arrow (RED = fly-through direction)
            ax.annotate('',
                xy=(pos[dim_a] + normal[dim_a] * arrow_len,
                    pos[dim_b] + normal[dim_b] * arrow_len),
                xytext=(pos[dim_a], pos[dim_b]),
                arrowprops=dict(arrowstyle='->', color='#ff4444',
                               lw=2.5, mutation_scale=18),
                zorder=8)
            
            # SEGMENT direction arrow (CYAN = progress direction)
            ax.annotate('',
                xy=(pos[dim_a] + seg_dir[dim_a] * arrow_len * 0.8,
                    pos[dim_b] + seg_dir[dim_b] * arrow_len * 0.8),
                xytext=(pos[dim_a], pos[dim_b]),
                arrowprops=dict(arrowstyle='->', color='#00ccff',
                               lw=2.0, mutation_scale=14),
                zorder=7)
            
            # Gate opening (perpendicular to normal, centered at gate)
            # Show the gate_width/2 opening
            gate_perp = np.array([-normal[dim_b], normal[dim_a], 0][:2])
            if np.linalg.norm(gate_perp) > 1e-6:
                gate_perp = gate_perp / np.linalg.norm(gate_perp)
                hw = gate_width / 2
                p1 = np.array([pos[dim_a], pos[dim_b]]) + gate_perp * hw
                p2 = np.array([pos[dim_a], pos[dim_b]]) - gate_perp * hw
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                        color=gate_color, linewidth=3, alpha=0.8, zorder=6)
            
            # Label
            ax.text(pos[dim_a] + 1.0, pos[dim_b] + 1.0,
                    f'G{i}\n{angle:.0f}° {status}',
                    fontsize=8, fontweight='bold', color=gate_color,
                    ha='left', va='bottom', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                             alpha=0.8, edgecolor=gate_color, linewidth=0.5))
        
        # Start position
        ax.plot(start_pos[dim_a], start_pos[dim_b], '*', color='#ffaa00',
                markersize=18, markeredgecolor='white', markeredgewidth=0.5,
                zorder=10)
        ax.text(start_pos[dim_a] + 1.2, start_pos[dim_b] + 1.2, 'START',
                fontsize=10, fontweight='bold', color='#ffaa00')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1, color='gray')
    
    # Legend
    legend_text = (
        'Red arrows = Gate Normal (R×[1,0,0]) = fly-through direction\n'
        'Cyan arrows = Segment direction (progress axis)\n'
        'Colored bars = Gate opening (gate_width/2)\n'
        'Green/Yellow/Red = alignment quality (angle between normal & segment)\n'
        'Angle = misalignment between trigger plane and gate plane'
    )
    fig.text(0.5, 0.01, legend_text, ha='center', fontsize=10, color='#ccc',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#111', alpha=0.9,
                      edgecolor='#444'))
    
    fig.suptitle('Gate Collision Diagnostic — Normal vs Segment Alignment',
                 fontsize=15, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__),
                           'gate_collision_diagnostic.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✓ Saved diagnostic plot: {out_path}")


if __name__ == "__main__":
    main()
