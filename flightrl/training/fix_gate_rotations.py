#!/usr/bin/env python3
"""
Fix gate rotations so that each gate's normal (fly-through direction)
aligns with the incoming segment direction from the previous gate/start.

The gate normal is defined as R(q) * [1,0,0] in the C++ code.
We compute a quaternion for each gate such that its normal points along
the direction from the previous waypoint to this gate (the segment direction).
"""
import numpy as np
import ruamel.yaml
import sys
import os


def direction_to_quat(direction):
    """
    Compute quaternion that rotates [1,0,0] to the given direction.
    
    The gate normal = R(q) * [1,0,0], so we need a rotation that maps
    the X-axis to 'direction'. We keep the gate upright (Z-up) where possible.
    
    Returns [w, x, y, z] quaternion.
    """
    d = direction / np.linalg.norm(direction)
    
    # Target: gate_normal (x-axis of gate frame) = d
    # Gate up (z-axis of gate frame) should be as close to world-Z as possible
    # Gate right (y-axis) = up x normal (right-hand rule)
    
    world_up = np.array([0.0, 0.0, 1.0])
    
    # If direction is nearly vertical, use a different reference
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    
    # Gate right = world_up x gate_normal (then normalize)
    right = np.cross(world_up, d)
    right = right / np.linalg.norm(right)
    
    # Gate up = gate_normal x gate_right 
    up = np.cross(d, right)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix: columns are [normal, right, up]
    R = np.column_stack([d, right, up])
    
    # Convert rotation matrix to quaternion
    # Using Shepperd's method for numerical stability
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
    q = q / np.linalg.norm(q)  # normalize
    
    # Ensure w > 0 for canonical form
    if q[0] < 0:
        q = -q
    
    return q


def quat_to_R(w, x, y, z):
    """Quaternion to rotation matrix."""
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def main():
    config_path = os.path.expanduser("~/flightmare/flightlib/configs/quadrotor_env.yaml")
    
    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)
    
    start_pos = np.array(cfg["start_pos"], dtype=np.float64)
    gates_cfg = cfg["gates"]
    
    positions = []
    for g in gates_cfg:
        positions.append(np.array(g["position"], dtype=np.float64))
    
    print(f"Start position: {start_pos}")
    print(f"Number of gates: {len(positions)}")
    print()
    
    # For each gate, compute segment direction and new quaternion
    new_quats = []
    for i, pos in enumerate(positions):
        if i == 0:
            prev = start_pos
        else:
            prev = positions[i - 1]
        
        seg_dir = pos - prev
        seg_len = np.linalg.norm(seg_dir)
        seg_dir_normalized = seg_dir / seg_len
        
        q = direction_to_quat(seg_dir_normalized)
        new_quats.append(q)
        
        # Verify: R(q) * [1,0,0] should equal seg_dir_normalized
        R = quat_to_R(*q)
        normal = R[:, 0]
        dot = np.dot(normal, seg_dir_normalized)
        angle_err = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
        
        # Also show old normal for comparison
        old_q = gates_cfg[i]["rotation_quat"]
        R_old = quat_to_R(float(old_q[0]), float(old_q[1]), float(old_q[2]), float(old_q[3]))
        old_normal = R_old[:, 0]
        old_dot = np.dot(old_normal, seg_dir_normalized)
        old_angle = np.degrees(np.arccos(np.clip(abs(old_dot), 0, 1)))
        
        print(f"Gate {i}: pos={pos}")
        print(f"  Segment dir: {seg_dir_normalized}")
        print(f"  OLD normal:  {old_normal}  (angle to seg: {old_angle:.1f}°)")
        print(f"  NEW normal:  {normal}      (angle to seg: {angle_err:.1f}°)")
        print(f"  NEW quat: [{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}]")
        print()
    
    # Update the YAML config
    for i, q in enumerate(new_quats):
        gates_cfg[i]["rotation_quat"] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"✓ Updated {config_path} with new gate rotations")


if __name__ == "__main__":
    main()
