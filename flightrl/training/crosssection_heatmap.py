#!/usr/bin/env python3
"""
Cross-sectional heatmap of the radial reward around the spline centerline.
Shows concentric rings of reward intensity with the spline at center.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ── Config (matches current YAML) ──
GATE_WIDTH = 5.0
THRESHOLD = GATE_WIDTH / 2.0  # 2.5m
DEV_RATIO = 1.5
DEV_COEFF = 0.3  # new linear coeff

# For the cross-section, we compute s(p) at a fixed arc position.
# At the midpoint of the track (arc ~12m), s(p) = arc_to_nearest - ratio * dist
# When the drone is at radial offset r from the spline center,
# dist ≈ |r| and arc_to_nearest stays roughly constant.
# So s(p) ≈ arc_mid - ratio * |r|
# And deviation_penalty = -coeff * max(0, |r| - threshold)  [linear]
ARC_MID = 12.0  # midpoint arc length (used as reference)

def compute_reward_at_offset(x, y, arc_ref=ARC_MID):
    """Compute total reward potential at offset (x,y) from spline center."""
    r = np.sqrt(x**2 + y**2)
    # spline_progress component
    progress = arc_ref - DEV_RATIO * r
    # per-step deviation penalty (linear beyond threshold)
    excess = np.maximum(0, r - THRESHOLD)
    dev_penalty = -DEV_COEFF * excess
    return progress, dev_penalty, progress + dev_penalty

out_dir = os.path.dirname(os.path.abspath(__file__))
res = 600
extent = 12  # meters in each direction
x = np.linspace(-extent, extent, res)
y = np.linspace(-extent, extent, res)
X, Y = np.meshgrid(x, y)

progress_grid, devpen_grid, total_grid = compute_reward_at_offset(X, Y)

# ── Custom colormap: dark red → orange → green → bright green ──
cmap_reward = LinearSegmentedColormap.from_list('reward', [
    (0.0,  '#1a0000'),
    (0.15, '#8b0000'),
    (0.3,  '#cc3300'),
    (0.45, '#e67300'),
    (0.55, '#cccc00'),
    (0.7,  '#66cc33'),
    (0.85, '#33aa33'),
    (1.0,  '#00ff88'),
])

fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), facecolor='#0a0a14')
fig.suptitle("Cross-Sectional Reward Heatmap  —  View Perpendicular to Spline",
             fontsize=16, fontweight='bold', color='white', y=0.97)

panels = [
    ("spline_progress(p)\narc − 1.5 × |offset|", progress_grid),
    ("deviation penalty\n−0.3 × max(0, |offset| − 2.5)", devpen_grid),
    ("total reward potential\nprogress + deviation_penalty", total_grid),
]

for ax_i, (title, grid) in enumerate(panels):
    ax = axes[ax_i]
    ax.set_facecolor('#0a0a14')

    vmin, vmax = np.percentile(grid, [2, 98])
    if ax_i == 1:  # deviation penalty is ≤0
        vmin = np.percentile(grid, 1)
        vmax = 0.01

    im = ax.imshow(grid, extent=[-extent, extent, -extent, extent],
                   origin='lower', cmap=cmap_reward, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')

    # Gate width circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(THRESHOLD * np.cos(theta), THRESHOLD * np.sin(theta),
            '--', color='cyan', lw=2, alpha=0.8,
            label=f'gate_width/2 = {THRESHOLD}m')

    # Center crosshair
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.axvline(0, color='white', lw=0.5, alpha=0.3)
    ax.plot(0, 0, '+', color='white', ms=15, mew=2, zorder=10)

    # Distance rings
    for r_ring in [5, 7.5, 10]:
        ax.plot(r_ring * np.cos(theta), r_ring * np.sin(theta),
                ':', color='#ffffff', lw=0.5, alpha=0.2)
        ax.text(r_ring * 0.707 + 0.2, r_ring * 0.707 + 0.2,
                f'{r_ring}m', color='#666', fontsize=7, alpha=0.6)

    ax.set_xlabel("Lateral offset (m)", color='#c9d1d9', fontsize=10)
    ax.set_ylabel("Vertical offset (m)", color='#c9d1d9', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(colors='#8b949e')
    ax.legend(fontsize=8, loc='upper right', facecolor='#161b22',
              edgecolor='#30363d', labelcolor='white')

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("reward", color='#c9d1d9', fontsize=9)
    cb.ax.tick_params(colors='#8b949e')

# Annotate the center
axes[2].annotate('SPLINE\nCENTER', xy=(0, 0), xytext=(3.5, -4),
                 fontsize=10, color='white', fontweight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = os.path.join(out_dir, "reward_crosssection_heatmap.png")
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Saved: {out_path}")

# ── Also make a single large "total" heatmap with contour lines ──
fig2, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a14')
ax.set_facecolor('#0a0a14')

vmin, vmax = np.percentile(total_grid, [2, 98])
im = ax.imshow(total_grid, extent=[-extent, extent, -extent, extent],
               origin='lower', cmap=cmap_reward, vmin=vmin, vmax=vmax,
               interpolation='bilinear')

# Contour lines
levels = np.linspace(vmin, vmax, 15)
cs = ax.contour(X, Y, total_grid, levels=levels, colors='white', linewidths=0.4, alpha=0.3)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f', colors='white')

# Gate boundary
ax.plot(THRESHOLD * np.cos(theta), THRESHOLD * np.sin(theta),
        '--', color='cyan', lw=2.5, alpha=0.9, label=f'gate radius = {THRESHOLD}m')

# Center
ax.plot(0, 0, '+', color='white', ms=20, mew=3, zorder=10)

# Distance rings
for r_ring in [5, 7.5, 10]:
    ax.plot(r_ring * np.cos(theta), r_ring * np.sin(theta),
            ':', color='white', lw=0.6, alpha=0.25)

ax.set_xlabel("Lateral offset from spline (m)", color='#c9d1d9', fontsize=12)
ax.set_ylabel("Vertical offset from spline (m)", color='#c9d1d9', fontsize=12)
ax.set_title("Total Reward Potential — Cross-Section at Track Midpoint\n"
             f"s(p) = (arc − {DEV_RATIO}×|r|) − {DEV_COEFF}×max(0, |r| − {THRESHOLD})",
             color='white', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.tick_params(colors='#8b949e')
ax.legend(fontsize=10, loc='upper right', facecolor='#161b22',
          edgecolor='#30363d', labelcolor='white')

cb = fig2.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cb.set_label("s(p) reward potential", color='#c9d1d9', fontsize=11)
cb.ax.tick_params(colors='#8b949e')

# Annotations
ax.annotate('FREE ZONE\n(no per-step penalty)', xy=(0, 1.2), fontsize=9,
            color='cyan', ha='center', fontweight='bold')
ax.annotate('LINEAR\nPENALTY\nZONE', xy=(5.5, 0), fontsize=9,
            color='#ff6666', ha='center', fontweight='bold')

out_path2 = os.path.join(out_dir, "reward_crosssection_large.png")
fig2.savefig(out_path2, dpi=180, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"✓ Saved: {out_path2}")
