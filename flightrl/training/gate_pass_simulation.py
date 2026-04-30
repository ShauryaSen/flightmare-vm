#!/usr/bin/env python3
"""
Realistic gate-pass simulation — simulates a drone flying through
gates along realistic approach trajectories and checks whether the
C++ gate-pass detection correctly detects the pass.

KEY FINDING: The issue is that segment_progress is clamped to 
[0, seg_len + 0.5], and the gate-pass check requires s_prev < seg_len
AND s_curr >= seg_len. If the drone overshoots or approaches from
a non-segment direction, it may cross seg_len at the wrong location,
or worse, never transition past seg_len at all.
"""

import numpy as np
import os

gate_positions = np.array([
    [17.25, 3.63, 6.31], [16.75, 39.57, 6.76], [2.33, 27.87, 2.55],
    [2.20, 9.00, 1.99], [-7.31, -12.14, 3.23], [-7.61, -28.47, 2.64],
    [-0.01, -33.91, 2.10], [5.87, -28.44, 2.57], [7.25, -11.79, 2.55],
    [-10.23, 7.77, 2.02], [-10.05, 30.62, 2.90],
])
start_pos = np.array([17.5, -15.0, 6.31])
num_gates = len(gate_positions)
gate_width = 3.0
sim_dt = 0.02

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


def quat_to_R(w, x, y, z):
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def get_gate_normal(i):
    R = quat_to_R(*quats[i])
    n = R[:, 0]
    return n / np.linalg.norm(n)


def get_segment(gate_idx):
    if gate_idx == 0:
        g_prev = start_pos
    else:
        g_prev = gate_positions[(gate_idx - 1 + num_gates) % num_gates]
    return g_prev, gate_positions[gate_idx]


def segment_progress(pos, gate_idx):
    g1, g2 = get_segment(gate_idx)
    seg = g2 - g1
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6: return 0.0
    raw = np.dot(pos - g1, seg) / seg_len
    return max(0.0, min(raw, seg_len + 0.5))


def compute_d_p(pos, gate_idx):
    g = gate_positions[gate_idx]
    n = get_gate_normal(gate_idx)
    d_n = np.dot(pos - g, n)
    d_p = np.linalg.norm(pos - g - d_n * n)
    return d_p, d_n


def simulate_approach(gate_idx, speed_mps, approach_dir, lateral_offset=0.0):
    """
    Simulate a drone approaching a gate at given speed along approach_dir
    with optional lateral offset from the gate center.
    
    Returns: (passed, collided, missed, debug_info)
    """
    g_prev, g_curr = get_segment(gate_idx)
    seg = g_curr - g_prev
    seg_len = np.linalg.norm(seg)
    seg_dir = seg / seg_len
    normal = get_gate_normal(gate_idx)
    
    # Normalize approach direction
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    
    # Compute lateral offset direction (perpendicular to approach in XY)
    lateral_dir = np.cross(approach_dir, [0, 0, 1])
    if np.linalg.norm(lateral_dir) < 1e-6:
        lateral_dir = np.cross(approach_dir, [0, 1, 0])
    lateral_dir = lateral_dir / np.linalg.norm(lateral_dir)
    
    # Start 5m before gate center along approach direction
    start = g_curr - approach_dir * 5.0 + lateral_dir * lateral_offset
    
    pos = start.copy()
    vel = approach_dir * speed_mps
    
    s_prev = segment_progress(pos, gate_idx)
    
    passed = False
    collided = False
    missed = True
    debug = []
    
    for step in range(500):  # 10 seconds max
        pos += vel * sim_dt
        s_curr = segment_progress(pos, gate_idx)
        d_p, d_n = compute_d_p(pos, gate_idx)
        
        # Mirror the C++ logic exactly
        if s_prev < seg_len and s_curr >= seg_len:
            if d_p < gate_width / 2.0:
                passed = True
                missed = False
                debug.append(f"  PASS at step {step}: s={s_curr:.3f}, d_p={d_p:.3f}, d_n={d_n:.3f}")
                break
            else:
                collided = True
                missed = False
                debug.append(f"  COLLISION at step {step}: s={s_curr:.3f}, d_p={d_p:.3f}, d_n={d_n:.3f}")
                break
        else:
            s_prev = s_curr
        
        # Check if we've gone way past
        dist_to_gate = np.linalg.norm(pos - g_curr)
        if dist_to_gate > 10.0 and np.dot(pos - g_curr, approach_dir) > 0:
            debug.append(f"  MISSED (flew past) at step {step}: s={s_curr:.3f}, dist={dist_to_gate:.1f}")
            break
    
    return passed, collided, missed, debug


def main():
    print("=" * 90)
    print("  REALISTIC GATE-PASS SIMULATION")
    print("=" * 90)
    
    speeds = [3.0, 8.0, 15.0]  # m/s
    offsets = [0.0, 0.5, 1.0, 1.4]  # meters lateral offset
    
    for gate_idx in range(num_gates):
        g_prev, g_curr = get_segment(gate_idx)
        seg = g_curr - g_prev
        seg_dir = seg / np.linalg.norm(seg)
        normal = get_gate_normal(gate_idx)
        angle = np.degrees(np.arccos(np.clip(np.dot(seg_dir, normal), -1, 1)))
        
        print(f"\n  G{gate_idx} — seg↔normal angle: {angle:.1f}°")
        print(f"  {'Approach':>12} {'Speed':>6} {'Offset':>7} {'Result':>10} {'d_p':>8}")
        print(f"  {'-'*50}")
        
        for approach_name, approach_dir in [
            ("segment", seg_dir),
            ("normal", normal),
        ]:
            for speed in [8.0]:
                for offset in [0.0, 0.5, 1.0, 1.4]:
                    passed, collided, missed, debug = simulate_approach(
                        gate_idx, speed, approach_dir, offset)
                    
                    if passed:
                        result = "PASS ✓"
                    elif collided:
                        result = "COLLISION"
                    else:
                        result = "MISSED ✗"
                    
                    # Get d_p at detection point from debug
                    d_p_str = ""
                    for d in debug:
                        if "d_p=" in d:
                            d_p_str = d.split("d_p=")[1].split(",")[0]
                    
                    print(f"  {approach_name:>12} {speed:>5.0f} {offset:>6.1f}m  "
                          f"{result:>10} {d_p_str:>8}")
    
    # Key test: what if drone approaches from normal direction at speed?
    print(f"\n{'='*90}")
    print(f"  HIGH-SPEED APPROACH ANALYSIS")
    print(f"  Does the drone's speed cause it to 'skip' the trigger boundary?")
    print(f"{'='*90}")
    
    for gate_idx in [0, 1, 10]:  # Most misaligned gates
        g_prev, g_curr = get_segment(gate_idx)
        seg = g_curr - g_prev
        seg_dir = seg / np.linalg.norm(seg)
        seg_len = np.linalg.norm(seg)
        normal = get_gate_normal(gate_idx)
        
        print(f"\n  G{gate_idx}:")
        for speed in [3, 5, 8, 12, 20]:
            # Approach along normal direction
            start = g_curr - normal * 3.0
            pos = start.copy()
            vel = normal * speed
            s_prev = segment_progress(pos, gate_idx)
            
            crossed = False
            for step in range(500):
                pos += vel * sim_dt
                s_curr = segment_progress(pos, gate_idx)
                
                if s_prev < seg_len and s_curr >= seg_len:
                    d_p, d_n = compute_d_p(pos, gate_idx)
                    crossed = True
                    step_dist = speed * sim_dt
                    print(f"    {speed:>3} m/s: trigger at step {step}, "
                          f"d_p={d_p:.3f}, d_n={d_n:.3f}, "
                          f"step_size={step_dist:.3f}m, "
                          f"{'PASS' if d_p < gate_width/2 else 'COLLISION'}")
                    break
                s_prev = s_curr
            
            if not crossed:
                print(f"    {speed:>3} m/s: NEVER TRIGGERED (s stays clamped)")


if __name__ == "__main__":
    main()
