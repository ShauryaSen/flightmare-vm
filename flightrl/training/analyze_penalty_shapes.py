#!/usr/bin/env python3
"""
Compare different per-step lateral deviation penalty shapes:
  A) Current: quadratic beyond threshold  −c·max(0, d−T)²
  B) Linear beyond threshold              −c·max(0, d−T)
  C) Linear everywhere (no threshold)     −c·d
  D) Soft exponential                      −c·(exp(d/scale)−1) beyond T
  E) Capped quadratic                      −c·min(max(0,d−T)², cap)

Also computes the effective per-step reward magnitude at different
offsets to check whether the penalty dominates the progress signal.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ── Config from current YAML ──
GATE_WIDTH = 5.0
THRESHOLD = GATE_WIDTH / 2.0  # 2.5m
DEV_RATIO = 1.5
DEV_COEFF = 0.2
TOTAL_ARC = 24.1  # meters (from our spline)
SIM_DT = 0.02
# Rough estimates for scale comparison
TYPICAL_SPEED = 5.0  # m/s — moderate drone speed
STEPS_PER_SEC = 1.0 / SIM_DT  # 50
PROGRESS_PER_STEP = TYPICAL_SPEED * SIM_DT  # 0.1m of arc per step

out_dir = os.path.dirname(os.path.abspath(__file__))
dist = np.linspace(0, 12, 500)

# ═══════════════════════════════════════════════════════════════
# Define penalty shapes (all return negative values)
# ═══════════════════════════════════════════════════════════════

def penalty_A(d, coeff=DEV_COEFF, T=THRESHOLD):
    """Current: quadratic beyond threshold"""
    excess = np.maximum(0, d - T)
    return -coeff * excess**2

def penalty_B(d, coeff=DEV_COEFF, T=THRESHOLD):
    """Linear beyond threshold"""
    excess = np.maximum(0, d - T)
    return -coeff * excess

def penalty_C(d, coeff=DEV_COEFF):
    """Linear everywhere (no threshold)"""
    return -coeff * d

def penalty_D(d, coeff=DEV_COEFF, T=THRESHOLD, scale=2.0):
    """Soft exponential beyond threshold"""
    excess = np.maximum(0, d - T)
    return -coeff * (np.exp(excess / scale) - 1)

def penalty_E(d, coeff=DEV_COEFF, T=THRESHOLD, cap=25.0):
    """Capped quadratic beyond threshold"""
    excess = np.maximum(0, d - T)
    return -coeff * np.minimum(excess**2, cap)

penalties = {
    'A: Quadratic (current)': penalty_A,
    'B: Linear (threshold)': penalty_B,
    'C: Linear (everywhere)': penalty_C,
    'D: Soft exponential': penalty_D,
    'E: Capped quadratic': penalty_E,
}

colors = {
    'A: Quadratic (current)': '#f85149',
    'B: Linear (threshold)': '#58a6ff',
    'C: Linear (everywhere)': '#3fb950',
    'D: Soft exponential': '#d2a8ff',
    'E: Capped quadratic': '#f0883e',
}

# ═══════════════════════════════════════════════════════════════
# Plot 1: Penalty shapes comparison
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
fig.suptitle("Lateral Deviation Penalty Analysis", fontsize=18,
             fontweight='bold', color='white', y=0.98)

gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
              left=0.07, right=0.95, top=0.93, bottom=0.06)
ax_style = dict(facecolor='#161b22')

# ── Panel 1: Raw penalty values ──
ax1 = fig.add_subplot(gs[0, 0], **ax_style)
for name, fn in penalties.items():
    ax1.plot(dist, fn(dist), '-', color=colors[name], lw=2.5, label=name)
ax1.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7,
            label=f'threshold = {THRESHOLD}m')
ax1.axhline(0, color='#555', ls='-', lw=0.5)
ax1.set_xlabel("Lateral deviation from spline (m)", color='#c9d1d9', fontsize=11)
ax1.set_ylabel("Per-step penalty", color='#c9d1d9', fontsize=11)
ax1.set_title("Raw penalty per timestep (coeff=0.2)", color='white', fontsize=12)
ax1.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax1.tick_params(colors='#8b949e')
ax1.grid(True, alpha=0.15, color='#30363d')

# ── Panel 2: Gradient (derivative) ──
ax2 = fig.add_subplot(gs[0, 1], **ax_style)
dd = 0.01
for name, fn in penalties.items():
    grad = (fn(dist + dd) - fn(dist - dd)) / (2 * dd)
    ax2.plot(dist, grad, '-', color=colors[name], lw=2.5, label=name)
ax2.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7)
ax2.axhline(0, color='#555', ls='-', lw=0.5)
ax2.set_xlabel("Lateral deviation from spline (m)", color='#c9d1d9', fontsize=11)
ax2.set_ylabel("d(penalty)/d(dist)", color='#c9d1d9', fontsize=11)
ax2.set_title("Gradient of penalty (signal strength for learning)",
              color='white', fontsize=12)
ax2.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax2.tick_params(colors='#8b949e')
ax2.grid(True, alpha=0.15, color='#30363d')

# ── Panel 3: Penalty relative to progress signal ──
ax3 = fig.add_subplot(gs[1, 0], **ax_style)
for name, fn in penalties.items():
    ratio = np.abs(fn(dist)) / PROGRESS_PER_STEP
    ax3.plot(dist, ratio, '-', color=colors[name], lw=2.5, label=name)
ax3.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7)
ax3.axhline(1.0, color='white', ls=':', lw=1, alpha=0.5, label='penalty = progress')
ax3.axhline(5.0, color='#f85149', ls=':', lw=1, alpha=0.5, label='penalty = 5× progress')
ax3.set_xlabel("Lateral deviation from spline (m)", color='#c9d1d9', fontsize=11)
ax3.set_ylabel("|penalty| / progress_per_step", color='#c9d1d9', fontsize=11)
ax3.set_title(f"Penalty magnitude vs forward progress signal\n"
              f"(assuming {TYPICAL_SPEED} m/s → {PROGRESS_PER_STEP}m progress/step)",
              color='white', fontsize=11)
ax3.set_ylim(0, 30)
ax3.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax3.tick_params(colors='#8b949e')
ax3.grid(True, alpha=0.15, color='#30363d')

# ── Panel 4: Cumulative penalty over 1 second at constant offset ──
ax4 = fig.add_subplot(gs[1, 1], **ax_style)
for name, fn in penalties.items():
    cumulative = fn(dist) * STEPS_PER_SEC  # 50 steps
    ax4.plot(dist, cumulative, '-', color=colors[name], lw=2.5, label=name)
# Also show total progress earned in 1 second
progress_1sec = PROGRESS_PER_STEP * STEPS_PER_SEC  # 5m of arc = 5 units progress
ax4.axhline(-progress_1sec, color='#3fb950', ls=':', lw=2, alpha=0.8,
            label=f'1 sec of forward progress ({progress_1sec:.1f})')
ax4.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7)
ax4.set_xlabel("Lateral deviation from spline (m)", color='#c9d1d9', fontsize=11)
ax4.set_ylabel("Cumulative penalty over 1 second", color='#c9d1d9', fontsize=11)
ax4.set_title("Penalty accumulated in 1 second at constant offset\n"
              "(green dashed = 1 sec of forward progress for comparison)",
              color='white', fontsize=11)
ax4.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax4.tick_params(colors='#8b949e')
ax4.grid(True, alpha=0.15, color='#30363d')

# ── Panel 5: Total reward (spline_progress delta + penalty) for moving forward ──
ax5 = fig.add_subplot(gs[2, 0], **ax_style)
# Scenario: drone moves forward 0.1m along arc at each lateral offset
# The delta from spline_progress alone is approximately:
# Δarc ≈ PROGRESS_PER_STEP, Δdist ≈ 0 (moving parallel)
# So progress_delta ≈ PROGRESS_PER_STEP (positive forward reward)
forward_reward = PROGRESS_PER_STEP  # ~0.1
for name, fn in penalties.items():
    total = forward_reward + fn(dist)
    ax5.plot(dist, total, '-', color=colors[name], lw=2.5, label=name)
ax5.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7)
ax5.axhline(0, color='white', ls=':', lw=1, alpha=0.5, label='break-even')
ax5.fill_between(dist, 0, forward_reward, alpha=0.1, color='#3fb950')
ax5.set_xlabel("Lateral deviation from spline (m)", color='#c9d1d9', fontsize=11)
ax5.set_ylabel("Net reward per step", color='#c9d1d9', fontsize=11)
ax5.set_title("Net per-step reward when moving forward at 5 m/s\n"
              "(positive = forward progress outweighs penalty)",
              color='white', fontsize=11)
ax5.set_ylim(-3, 0.3)
ax5.legend(fontsize=7, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax5.tick_params(colors='#8b949e')
ax5.grid(True, alpha=0.15, color='#30363d')

# ── Panel 6: "Where does progress become futile?" ──
ax6 = fig.add_subplot(gs[2, 1], **ax_style)
# At what offset does the penalty per step exceed the progress per step?
# (i.e., the drone gets negative reward even when flying perfectly forward)
breakeven_text = []
for name, fn in penalties.items():
    pen = np.abs(fn(dist))
    idx = np.searchsorted(pen, forward_reward)
    be = dist[idx] if idx < len(dist) else float('inf')
    breakeven_text.append(f"{name}: {be:.1f}m")
    ax6.barh(name.split(':')[0].strip(), be, color=colors[name], alpha=0.8, height=0.6)
    ax6.text(be + 0.1, list(penalties.keys()).index(name),
             f'{be:.1f}m', va='center', color='white', fontsize=11, fontweight='bold')

ax6.set_xlabel("Break-even lateral offset (m)", color='#c9d1d9', fontsize=11)
ax6.set_title("Distance where penalty = forward progress (at 5 m/s)\n"
              "Beyond this, the drone gets negative reward even when flying forward",
              color='white', fontsize=11)
ax6.axvline(THRESHOLD, color='cyan', ls='--', lw=1.5, alpha=0.7, label=f'gate boundary ({THRESHOLD}m)')
ax6.legend(fontsize=9, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax6.tick_params(colors='#8b949e')
ax6.grid(True, alpha=0.15, color='#30363d', axis='x')
ax6.set_xlim(0, 12)

out_path = os.path.join(out_dir, "penalty_shape_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Saved: {out_path}")

# ═══════════════════════════════════════════════════════════════
# Print numerical analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("NUMERICAL ANALYSIS: Penalty vs Progress at key distances")
print("="*70)
print(f"\nConfig: deviation_ratio={DEV_RATIO}, coeff={DEV_COEFF}, "
      f"gate_width={GATE_WIDTH}, threshold={THRESHOLD}m")
print(f"Progress per step at {TYPICAL_SPEED} m/s: {PROGRESS_PER_STEP:.3f}")
print(f"Steps per second: {STEPS_PER_SEC:.0f}")

checkpoints = [0, 1, 2, 2.5, 3, 4, 5, 7.5, 10]
print(f"\n{'Offset':>8s} | {'A:Quad':>10s} | {'B:Lin-T':>10s} | "
      f"{'C:Lin-All':>10s} | {'D:Exp':>10s} | {'E:Cap-Q':>10s} | "
      f"{'A/prog':>8s}")
print("-" * 90)
for d in checkpoints:
    vals = [fn(np.array([d]))[0] for fn in penalties.values()]
    ratio = abs(vals[0]) / PROGRESS_PER_STEP if PROGRESS_PER_STEP > 0 else 0
    print(f"{d:>7.1f}m | {vals[0]:>10.4f} | {vals[1]:>10.4f} | "
          f"{vals[2]:>10.4f} | {vals[3]:>10.4f} | {vals[4]:>10.4f} | "
          f"{ratio:>7.1f}×")

print("\n" + "="*70)
print("KEY INSIGHT: Built-in penalty from spline_progress delta")
print("="*70)
print(f"""
Remember: the reward per step is:
  reward = [s(p_curr) - s(p_prev)] + per_step_penalty

Where s(p) = arc_progress - {DEV_RATIO} × dist

So the delta ALREADY contains: -{DEV_RATIO} × Δ(dist)
  → Moving 1m further from track costs {DEV_RATIO} reward from the delta alone
  → But staying at CONSTANT offset has Δ(dist)=0, so no penalty from this term!

The per-step penalty exists to handle the constant-offset case.

PROBLEM WITH CURRENT QUADRATIC (A):
  At 5m offset (2.5m excess): penalty = -0.2 × 2.5² = {penalty_A(np.array([5.0]))[0]:.3f}/step
  Forward progress at 5 m/s:  +{PROGRESS_PER_STEP:.3f}/step
  Ratio: {abs(penalty_A(np.array([5.0]))[0]) / PROGRESS_PER_STEP:.1f}× ← penalty dominates!

  The drone sees that being 5m off-track makes forward progress FUTILE.
  This teaches "avoid being far" but can suppress "make forward progress."

WITH LINEAR BEYOND THRESHOLD (B):
  At 5m offset (2.5m excess): penalty = -0.2 × 2.5 = {penalty_B(np.array([5.0]))[0]:.3f}/step
  Forward progress at 5 m/s:  +{PROGRESS_PER_STEP:.3f}/step
  Ratio: {abs(penalty_B(np.array([5.0]))[0]) / PROGRESS_PER_STEP:.1f}× ← still significant but manageable

RECOMMENDATION:
  The linear-beyond-threshold (B) with a higher coefficient might be the sweet
  spot: it provides constant "pull" back toward the track without the harsh
  quadratic cliff that can dominate the progress signal at moderate distances.
""")

if __name__ == "__main__":
    pass
