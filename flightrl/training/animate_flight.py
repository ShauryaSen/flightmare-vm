#!/usr/bin/env python3
"""Animate the trajectory controller flight from saved demo data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

# ─── Track definition ──────────────────────────────────────────
RADIUS = 18.0
Z_TRACK = 3.0
N_GATES = 8
START_POS = np.array([18.0, -4.0, Z_TRACK])
GATE_POSITIONS = []
for i in range(N_GATES):
    theta = i * 2 * np.pi / N_GATES
    GATE_POSITIONS.append([RADIUS * np.cos(theta), RADIUS * np.sin(theta), Z_TRACK])
GATE_POSITIONS = np.array(GATE_POSITIONS)

SIM_DT = 0.02

def main():
    # Load demo data
    data = np.load('demos/circle_8_demos.npz')
    positions = data['positions']
    ref_positions = data['ref_positions']
    rewards = data['rewards']
    observations = data['observations']
    
    # Extract velocities and rotation from observations
    velocities = observations[:, 0:3]
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Subsample for smoother animation (every 2 steps = 25fps)
    skip = 2
    pos = positions[::skip]
    ref = ref_positions[::skip]
    spd = speeds[::skip]
    rew = rewards[::skip]
    n_frames = len(pos)
    
    # ── Figure setup ──
    fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')
    
    # Top-down view (main)
    ax1 = fig.add_subplot(1, 2, 1, facecolor='#16213e')
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)', color='white', fontsize=10)
    ax1.set_ylabel('Y (m)', color='white', fontsize=10)
    ax1.set_title('Top-Down View', color='white', fontsize=13, fontweight='bold')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    
    # Draw track circle
    theta_circle = np.linspace(0, 2*np.pi, 200)
    ax1.plot(RADIUS*np.cos(theta_circle), RADIUS*np.sin(theta_circle), 
             color='#333', linestyle='--', linewidth=1, alpha=0.5)
    
    # Draw gates
    gate_width = 2.5  # visual width
    gate_rotations_deg = [90, 135, 180, 225, 270, 315, 0, 45]
    for i, gp in enumerate(GATE_POSITIONS):
        angle = np.radians(gate_rotations_deg[i])
        dx = gate_width * np.cos(angle + np.pi/2)
        dy = gate_width * np.sin(angle + np.pi/2)
        ax1.plot([gp[0]-dx, gp[0]+dx], [gp[1]-dy, gp[1]+dy], 
                 color='#e94560', linewidth=3, solid_capstyle='round')
        ax1.text(gp[0]+1.5, gp[1]+1.5, f'G{i}', color='#e94560', 
                 fontsize=8, fontweight='bold', alpha=0.8)
    
    # Reference trajectory (faint)
    ax1.plot(ref[:, 0], ref[:, 1], color='#0f3460', linewidth=1, alpha=0.3)
    
    # Animated elements
    trail_line, = ax1.plot([], [], color='#00d2ff', linewidth=1.5, alpha=0.6)
    drone_dot, = ax1.plot([], [], 'o', color='#00ff88', markersize=8, 
                          markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ref_dot, = ax1.plot([], [], 'x', color='#ffaa00', markersize=6, alpha=0.5)
    
    # Side view (Z profile)
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#16213e')
    ax2.set_xlim(0, n_frames * skip * SIM_DT)
    ax2.set_ylim(0, 6)
    ax2.set_xlabel('Time (s)', color='white', fontsize=9)
    ax2.set_ylabel('Altitude (m)', color='white', fontsize=9)
    ax2.set_title('Altitude Profile', color='white', fontsize=11, fontweight='bold')
    ax2.axhline(Z_TRACK, color='#333', linestyle='--', alpha=0.5)
    ax2.tick_params(colors='white', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#444')
    
    z_line, = ax2.plot([], [], color='#00d2ff', linewidth=1.5)
    z_ref_line, = ax2.plot([], [], color='#ffaa00', linewidth=1, alpha=0.4)
    
    # Speed + reward view
    ax3 = fig.add_subplot(2, 2, 4, facecolor='#16213e')
    ax3.set_xlim(0, n_frames * skip * SIM_DT)
    ax3.set_ylim(-1, 8)
    ax3.set_xlabel('Time (s)', color='white', fontsize=9)
    ax3.set_ylabel('Speed (m/s)', color='white', fontsize=9)
    ax3.set_title('Speed & Rewards', color='white', fontsize=11, fontweight='bold')
    ax3.tick_params(colors='white', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color('#444')
    
    speed_line, = ax3.plot([], [], color='#ff6b6b', linewidth=1.5, label='Speed')
    reward_scatter = ax3.scatter([], [], c='#00ff88', s=20, zorder=5, label='Gate reward')
    ax3.legend(loc='upper right', fontsize=7, facecolor='#16213e', 
               edgecolor='#444', labelcolor='white')
    
    # Status text
    status_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                           color='white', fontsize=9, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.8))
    
    # Gate counter
    gates_passed = 0
    
    def init():
        trail_line.set_data([], [])
        drone_dot.set_data([], [])
        ref_dot.set_data([], [])
        z_line.set_data([], [])
        z_ref_line.set_data([], [])
        speed_line.set_data([], [])
        status_text.set_text('')
        return trail_line, drone_dot, ref_dot, z_line, z_ref_line, speed_line, status_text
    
    def animate(frame):
        nonlocal gates_passed
        
        # Trail (last 100 frames)
        trail_start = max(0, frame - 100)
        trail_line.set_data(pos[trail_start:frame+1, 0], pos[trail_start:frame+1, 1])
        
        # Current position
        drone_dot.set_data([pos[frame, 0]], [pos[frame, 1]])
        ref_dot.set_data([ref[frame, 0]], [ref[frame, 1]])
        
        # Time arrays
        t_arr = np.arange(frame+1) * skip * SIM_DT
        
        # Altitude
        z_line.set_data(t_arr, pos[:frame+1, 2])
        z_ref_line.set_data(t_arr, ref[:frame+1, 2])
        
        # Speed
        speed_line.set_data(t_arr, spd[:frame+1])
        
        # Gate rewards (show as scatter points where reward > 2)
        gate_mask = rew[:frame+1] > 2.0
        if np.any(gate_mask):
            gate_times = t_arr[gate_mask]
            gate_speeds = spd[:frame+1][gate_mask]
            reward_scatter.set_offsets(np.column_stack([gate_times, gate_speeds]))
            gates_passed = int(np.sum(gate_mask))
        
        # Check for gate proximity (highlight nearest gate)
        dists_to_gates = np.linalg.norm(GATE_POSITIONS[:, :2] - pos[frame, :2], axis=1)
        nearest_gate = np.argmin(dists_to_gates)
        nearest_dist = dists_to_gates[nearest_gate]
        
        # Tracking error
        err = np.linalg.norm(pos[frame] - ref[frame])
        
        # Status
        t_cur = frame * skip * SIM_DT
        status_text.set_text(
            f't={t_cur:5.1f}s  gates={gates_passed}/8\n'
            f'speed={spd[frame]:.1f} m/s\n'
            f'alt={pos[frame,2]:.2f}m  err={err:.2f}m\n'
            f'nearest: G{nearest_gate} ({nearest_dist:.1f}m)'
        )
        
        return trail_line, drone_dot, ref_dot, z_line, z_ref_line, speed_line, status_text, reward_scatter
    
    print(f"Creating animation with {n_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=n_frames, interval=40,  # 25fps
                                    blit=False)
    
    plt.tight_layout()
    
    # Save as GIF
    out_path = 'flight_animation.gif'
    print(f"Saving to {out_path}...")
    anim.save(out_path, writer='pillow', fps=25, dpi=100)
    print(f"Done! Saved {out_path}")

if __name__ == '__main__':
    main()
