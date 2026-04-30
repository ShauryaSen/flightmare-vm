#!/usr/bin/env python3
"""
Diagnose the ACTUAL failure mode: 
- Is the drone hitting the trigger plane at d_p > gate_width/2?
- What's the gate_width/2 = 1.5m really mean physically?

The drone approaches gates with non-zero lateral error because it's learning.
A beginning RL agent won't fly straight through the center. Let's see how
forgiving the gate_width/2 = 1.5m window actually is.

Also checks: does the trigger fire reliably, or does the discrete timestep
cause the drone to "skip over" the trigger zone?
"""

import numpy as np

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


def main():
    print("=" * 90)
    print("  GATE OPENING SIZE ANALYSIS")
    print("=" * 90)
    
    print(f"\n  gate_width = {gate_width}m → opening radius = {gate_width/2}m")
    print(f"  AlphaPilot physical gates are ~1.2m wide")
    print(f"  Current gate_width/2 = 1.5m means the drone must pass within 1.5m of center")
    print()
    
    # For a beginning RL agent, what lateral error is typical?
    # At speed v, sim_dt=0.02, with random torques, the drone can easily
    # deviate 2-5 meters from ideal path near a gate.
    
    # Let's compute: at the moment of trigger, what's the maximum d_p
    # for a drone that's been moving at an angle?
    
    print(f"  KEY QUESTION: When the trigger fires (s >= seg_len), how big")
    print(f"  is the 'capture zone' where d_p < gate_width/2?")
    print(f"  This is the effective gate opening as seen by the RL agent.")
    print()
    
    # For each gate, compute the effective opening in physical space
    for i in range(num_gates):
        g_prev, g_curr = get_segment(i)
        seg = g_curr - g_prev
        seg_len = np.linalg.norm(seg)
        seg_dir = seg / seg_len
        normal = get_gate_normal(i)
        
        # The trigger plane is: dot(pos - g_prev, seg_dir) = seg_len
        # This is the plane perpendicular to seg_dir, passing through g_curr
        
        # d_p = |pos - g_curr - d_n * normal|
        # At the trigger plane, pos is constrained to:
        #   pos = g_curr + t1*e1 + t2*e2  (for any t1, t2 in the plane)
        # where e1, e2 are basis vectors of the plane ⊥ seg_dir
        
        # But WAIT: the trigger fires at s_curr >= seg_len, which means
        # pos is on the plane dot(pos - g_prev, seg_dir) = seg_len
        # which is dot(pos - g_curr, seg_dir) = 0
        # So pos - g_curr is perpendicular to seg_dir.
        
        # d_p depends on the projection of (pos - g_curr) onto the
        # plane perpendicular to normal. Let's compute this.
        
        # Actually, wait. At the trigger plane:
        #   pos = g_curr + t * perp   (where perp ⊥ seg_dir)
        # Then:
        #   d_n = dot(pos - g_curr, normal) = t * dot(perp, normal)
        #   d_p = |pos - g_curr - d_n * normal| = |t*perp - d_n*normal|
        
        # For t = 0 (dead center): d_p = 0 ✓
        # For t = 1m offset: d_p depends on which perpendicular direction
        
        # The worst case is when perp is parallel to normal → d_p = 0 
        # (the offset is entirely along the normal axis, doesn't count)
        # The best case is when perp ⊥ normal → d_p = t
        
        # The angle between seg_dir and normal determines how much of
        # a lateral offset (in the trigger plane) projects onto d_p.
        
        angle = np.degrees(np.arccos(np.clip(abs(np.dot(seg_dir, normal)), 0, 1)))
        
        # Effective opening radius depends on the angle
        # When angle=0: trigger plane = gate plane, opening = gate_width/2
        # When angle=90: trigger plane ⊥ gate plane, opening = ∞ along one axis
        
        print(f"  G{i}: seg↔normal angle = {angle:.1f}°, seg_len = {seg_len:.1f}m")
    
    print(f"\n{'='*90}")
    print(f"  MONTE CARLO: Random approaches to each gate")
    print(f"  Testing 1000 random drone positions near each gate")
    print(f"{'='*90}")
    
    np.random.seed(42)
    
    for i in range(num_gates):
        g_prev, g_curr = get_segment(i)
        seg = g_curr - g_prev
        seg_len = np.linalg.norm(seg)
        seg_dir = seg / seg_len
        normal = get_gate_normal(i)
        
        # Generate random positions near the trigger plane
        n_samples = 5000
        pass_count = 0
        collision_count = 0
        miss_count = 0
        
        for _ in range(n_samples):
            # Random offset from gate center, uniform in [-3, 3]m in each axis
            offset = np.random.uniform(-3, 3, 3)
            
            # Project to near the trigger plane (within 1 step distance)
            pos_on_plane = g_curr + offset
            # Adjust to be on the trigger plane: dot(pos - g_prev, seg_dir) = seg_len
            correction = seg_len - np.dot(pos_on_plane - g_prev, seg_dir)
            pos_on_plane += correction * seg_dir
            
            # Check: s_prev would be seg_len - epsilon (just before trigger)
            s_curr = segment_progress(pos_on_plane, i)
            
            # d_p check
            d_n = np.dot(pos_on_plane - g_curr, normal)
            d_p = np.linalg.norm(pos_on_plane - g_curr - d_n * normal)
            
            if s_curr >= seg_len:
                if d_p < gate_width / 2:
                    pass_count += 1
                else:
                    collision_count += 1
            else:
                miss_count += 1
        
        total_triggered = pass_count + collision_count
        pass_rate = pass_count / max(total_triggered, 1) * 100
        
        print(f"  G{i}: pass={pass_count} ({pass_rate:.1f}%), "
              f"collision={collision_count}, no_trigger={miss_count}")
    
    print(f"\n{'='*90}")
    print(f"  CONCLUSION")
    print(f"{'='*90}")
    print(f"""
  With gate_width = {gate_width}m, the pass opening radius is {gate_width/2}m.
  This means the drone must be within 1.5m of the gate NORMAL AXIS 
  (not within 1.5m of the gate center!) at the exact moment the trigger fires.
  
  For a learning agent with noisy control:
  - Early training: lateral errors of 2-5m are common → ALWAYS collision
  - The gate opening is VERY unforgiving for an untrained agent
  
  RECOMMENDATION: 
  - During early training, use gate_width = 6-10m (very forgiving)
  - Curriculum: gradually reduce gate_width as the agent improves
  - Or: remove collision termination entirely during Phase 1
    """)


if __name__ == "__main__":
    main()
