#!/usr/bin/env python3
"""
Visualize the radial reward profile at cross-sections along the spline.

For each point along the spline, shows what spline_progress() returns as
the drone moves radially away from the track centerline. Generates:
  1) A static multi-panel PNG with heatmap + slices + decomposition
  2) An interactive HTML page with sliders for coefficient tuning

Matches the C++ reward math exactly (cubic Hermite spline + brute-force
nearest-sample projection).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ruamel.yaml
import os, json

# ═══════════════════════════════════════════════════════════════
# Cubic Hermite spline — exact port of C++ buildTrackSpline()
# ═══════════════════════════════════════════════════════════════

def quat_to_R(w, x, y, z):
    n = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def euler_to_R(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ]
    ])

def hermite_eval(p0, m0, p1, m1, t):
    t2, t3 = t*t, t*t*t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

SAMPLES_PER_SEG = 200

def load_config():
    cfg_path = os.path.expanduser("~/flightmare/flightlib/configs/quadrotor_env.yaml")
    yaml = ruamel.yaml.YAML()
    with open(cfg_path) as f:
        cfg = yaml.load(f)
    start_pos = np.array(cfg["start_pos"], dtype=np.float64)
    gates_pos = []
    gate_normals = []
    for g in cfg["gates"]:
        pos = np.array(g["position"], dtype=np.float64)
        gates_pos.append(pos)
        if "rotation_quat" in g:
            q = g["rotation_quat"]
            R = quat_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        else:
            rot = g["rotation"]
            R = euler_to_R(float(rot[0]), float(rot[1]), float(rot[2]))
        gate_normals.append(R[:, 0] / np.linalg.norm(R[:, 0]))
    rl = cfg["rl"]
    params = {
        "gate_width": float(rl.get("gate_width", 5)),
        "spline_deviation_ratio": float(rl.get("spline_deviation_ratio", 1.5)),
        "deviation_penalty_coeff": float(rl.get("deviation_penalty_coeff", 0.01)),
    }
    return start_pos, np.array(gates_pos), gate_normals, params

def build_spline(start_pos, gates_pos, gate_normals):
    num_gates = len(gates_pos)
    ctrl_pts = [start_pos] + [g for g in gates_pos]
    tangent_dirs = [gates_pos[0] - start_pos]
    tangent_dirs[0] /= max(np.linalg.norm(tangent_dirs[0]), 1e-8)
    for n in gate_normals:
        tangent_dirs.append(n)

    segments = []
    global_arc = 0.0
    for si in range(num_gates):
        p0, p1 = ctrl_pts[si], ctrl_pts[si + 1]
        chord = np.linalg.norm(p1 - p0)
        m0 = tangent_dirs[si] * chord
        m1 = tangent_dirs[si + 1] * chord
        samples = [p0.copy()]
        arc_lengths = [global_arc]
        s_start = global_arc
        for j in range(1, SAMPLES_PER_SEG):
            t = j / (SAMPLES_PER_SEG - 1)
            pt = hermite_eval(p0, m0, p1, m1, t)
            global_arc += np.linalg.norm(pt - samples[-1])
            samples.append(pt)
            arc_lengths.append(global_arc)
        segments.append({
            "s_start": s_start, "s_end": global_arc,
            "samples": np.array(samples), "arc_lengths": np.array(arc_lengths)
        })
    return segments, global_arc

def spline_progress_at(pos, seg, deviation_ratio):
    """Exact port of C++ spline_progress for a single segment."""
    diffs = seg["samples"] - pos[np.newaxis, :]
    dists_sq = np.sum(diffs**2, axis=1)
    best_idx = int(np.argmin(dists_sq))
    best_dist = np.sqrt(dists_sq[best_idx])
    best_arc = seg["arc_lengths"][best_idx]

    for offset in [-1, 1]:
        j2 = best_idx + offset
        if j2 < 0 or j2 >= len(seg["samples"]):
            continue
        a = seg["samples"][best_idx]
        b = seg["samples"][j2]
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 < 1e-12:
            continue
        t = np.clip(np.dot(pos - a, ab) / ab_len2, 0, 1)
        proj = a + t * ab
        d = np.linalg.norm(pos - proj)
        if d < best_dist:
            best_dist = d
            best_arc = seg["arc_lengths"][best_idx] + t * (seg["arc_lengths"][j2] - seg["arc_lengths"][best_idx])

    best_arc = np.clip(best_arc, seg["s_start"], seg["s_end"])
    progress = (best_arc - seg["s_start"]) - deviation_ratio * best_dist
    return progress, best_dist

def compute_normals_at_samples(segments):
    all_normals = []
    for seg in segments:
        samples = seg["samples"]
        normals = np.zeros_like(samples)
        for i in range(len(samples)):
            if i < len(samples) - 1:
                tangent = samples[min(i+1, len(samples)-1)] - samples[max(i-1, 0)]
            else:
                tangent = samples[i] - samples[i-1]
            tangent_2d = tangent[:2]
            n2d = np.array([-tangent_2d[1], tangent_2d[0]])
            n_len = np.linalg.norm(n2d)
            if n_len > 1e-8:
                n2d /= n_len
            normals[i] = np.array([n2d[0], n2d[1], 0.0])
        all_normals.append(normals)
    return all_normals

def compute_reward_landscape(segments, normals, params, arc_steps=150, radial_steps=120, max_radial=10.0):
    dev_ratio = params["spline_deviation_ratio"]
    dev_coeff = params["deviation_penalty_coeff"]
    gate_w = params["gate_width"]
    threshold = gate_w / 2.0

    total_arc = segments[-1]["s_end"]
    arc_positions = np.linspace(0, total_arc, arc_steps)
    radial_offsets = np.linspace(-max_radial, max_radial, radial_steps)

    spline_pts = []
    spline_normals_arr = []
    seg_indices = []
    for arc in arc_positions:
        for si, seg in enumerate(segments):
            if arc <= seg["s_end"] or si == len(segments) - 1:
                idx = np.searchsorted(seg["arc_lengths"], arc)
                idx = min(idx, len(seg["samples"]) - 1)
                spline_pts.append(seg["samples"][idx])
                spline_normals_arr.append(normals[si][idx])
                seg_indices.append(si)
                break

    spline_pts = np.array(spline_pts)
    spline_normals_arr = np.array(spline_normals_arr)

    progress_grid = np.zeros((arc_steps, radial_steps))
    deviation_grid = np.zeros((arc_steps, radial_steps))

    for i in range(arc_steps):
        seg = segments[seg_indices[i]]
        center = spline_pts[i]
        normal = spline_normals_arr[i]
        for j, r in enumerate(radial_offsets):
            pos = center + r * normal
            prog, dist = spline_progress_at(pos, seg, dev_ratio)
            progress_grid[i, j] = prog
            dev_pen = 0.0
            if dev_coeff > 0 and dist > threshold:
                excess = dist - threshold
                dev_pen = -dev_coeff * excess * excess
            deviation_grid[i, j] = dev_pen

    total_grid = progress_grid + deviation_grid
    return arc_positions, radial_offsets, progress_grid, deviation_grid, total_grid, spline_pts, seg_indices

# ═══════════════════════════════════════════════════════════════
# Static PNG
# ═══════════════════════════════════════════════════════════════

def render_static(segments, normals, params, start_pos, gates_pos, total_arc, out_dir):
    max_radial = 10.0
    arc_steps, radial_steps = 150, 120

    arc_pos, rad_off, prog_grid, dev_grid, total_grid, spline_pts, seg_indices = \
        compute_reward_landscape(segments, normals, params, arc_steps, radial_steps, max_radial)

    gate_w = params["gate_width"]

    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117')
    fig.suptitle("Reward Landscape: Radial Profile Along Spline",
                 fontsize=18, fontweight='bold', color='white', y=0.97)

    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 2.5],
                  hspace=0.3, wspace=0.3,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    ax_style = dict(facecolor='#161b22')

    # ── 1. Heatmap: total reward potential ──
    ax_heat = fig.add_subplot(gs[0, :2], **ax_style)
    extent = [arc_pos[0], arc_pos[-1], rad_off[0], rad_off[-1]]
    vmax = np.percentile(total_grid, 98)
    vmin = np.percentile(total_grid, 2)
    im = ax_heat.imshow(total_grid.T, aspect='auto', origin='lower', extent=extent,
                        cmap='RdYlGn', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax_heat.axhline(gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8, label=f'gate_width/2 = {gate_w/2:.1f}m')
    ax_heat.axhline(-gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8)
    ax_heat.axhline(0, color='white', ls='-', lw=0.8, alpha=0.4)
    for seg in segments:
        ax_heat.axvline(seg["s_end"], color='#888', ls=':', lw=1, alpha=0.5)
    ax_heat.set_xlabel("Arc length along spline (m)", color='#c9d1d9', fontsize=11)
    ax_heat.set_ylabel("Radial offset from centerline (m)", color='#c9d1d9', fontsize=11)
    ax_heat.set_title(
        f"s(p) = (arc_progress) − {params['spline_deviation_ratio']}×|lateral| "
        f"− {params['deviation_penalty_coeff']}×max(0, dev−{gate_w/2:.1f})²",
        color='#8b949e', fontsize=10)
    ax_heat.legend(fontsize=9, loc='upper right', facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax_heat.tick_params(colors='#8b949e')
    cb = fig.colorbar(im, ax=ax_heat, pad=0.01, shrink=0.9)
    cb.set_label("s(p) reward potential", color='#c9d1d9', fontsize=10)
    cb.ax.tick_params(colors='#8b949e')

    # ── 2. Track top-down view ──
    ax_track = fig.add_subplot(gs[0, 2], **ax_style)
    for si, seg in enumerate(segments):
        ax_track.plot(seg["samples"][:, 0], seg["samples"][:, 1],
                      '-', color='#FFD700', lw=2.5, alpha=0.9)
    ax_track.plot(*start_pos[:2], '*', color='lime', ms=16, zorder=10, label='Start')
    for i, gp in enumerate(gates_pos):
        ax_track.plot(*gp[:2], 'D', color='#58a6ff', ms=9, zorder=10)
        ax_track.annotate(f'G{i}', gp[:2], textcoords="offset points",
                          xytext=(7, 7), color='#58a6ff', fontsize=9, fontweight='bold')
    ax_track.set_xlabel("X (m)", color='#c9d1d9', fontsize=11)
    ax_track.set_ylabel("Y (m)", color='#c9d1d9', fontsize=11)
    ax_track.set_title("Track Layout (top-down)", color='white', fontsize=12)
    ax_track.set_aspect('equal')
    ax_track.tick_params(colors='#8b949e')
    ax_track.grid(True, alpha=0.15, color='#30363d')
    ax_track.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

    # ── 3. Cross-section slices ──
    ax_slice = fig.add_subplot(gs[1, :2], **ax_style)
    slice_arcs = np.linspace(arc_pos[0], arc_pos[-1], 7)[1:-1]
    colors_list = plt.cm.viridis(np.linspace(0.15, 0.95, len(slice_arcs)))
    for k, sa in enumerate(slice_arcs):
        idx = np.argmin(np.abs(arc_pos - sa))
        ax_slice.plot(rad_off, total_grid[idx, :], '-', color=colors_list[k],
                      lw=2, label=f'arc={sa:.1f}m')
    ax_slice.axvline(gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8)
    ax_slice.axvline(-gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8)
    ax_slice.axvline(0, color='white', ls='-', lw=0.8, alpha=0.4)
    ax_slice.axhline(0, color='#555', ls='-', lw=0.5)
    ax_slice.set_xlabel("Radial offset from centerline (m)", color='#c9d1d9', fontsize=11)
    ax_slice.set_ylabel("s(p) reward potential", color='#c9d1d9', fontsize=11)
    ax_slice.set_title("Cross-section reward profiles at different arc positions",
                       color='white', fontsize=12)
    ax_slice.legend(fontsize=8, loc='lower left', ncol=2,
                    facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax_slice.tick_params(colors='#8b949e')
    ax_slice.grid(True, alpha=0.15, color='#30363d')

    # ── 4. Decomposition at midpoint ──
    ax_decomp = fig.add_subplot(gs[1, 2], **ax_style)
    mid_idx = arc_steps // 2
    ax_decomp.plot(rad_off, prog_grid[mid_idx, :], '-', color='#3fb950', lw=2, label='spline_progress')
    ax_decomp.plot(rad_off, dev_grid[mid_idx, :], '-', color='#f85149', lw=2, label='deviation_penalty')
    ax_decomp.plot(rad_off, total_grid[mid_idx, :], '-', color='#d2a8ff', lw=2.5, label='total')
    ax_decomp.axvline(gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8)
    ax_decomp.axvline(-gate_w/2, color='cyan', ls='--', lw=1.5, alpha=0.8)
    ax_decomp.axhline(0, color='#555', ls='-', lw=0.5)
    ax_decomp.set_xlabel("Radial offset (m)", color='#c9d1d9', fontsize=11)
    ax_decomp.set_ylabel("Reward component", color='#c9d1d9', fontsize=11)
    ax_decomp.set_title(f"Reward decomposition at arc={arc_pos[mid_idx]:.1f}m",
                        color='white', fontsize=12)
    ax_decomp.legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax_decomp.tick_params(colors='#8b949e')
    ax_decomp.grid(True, alpha=0.15, color='#30363d')

    # ── Config info ──
    info = (f"spline_deviation_ratio={params['spline_deviation_ratio']}  "
            f"deviation_penalty_coeff={params['deviation_penalty_coeff']}  "
            f"gate_width={params['gate_width']}  "
            f"total_arc={total_arc:.1f}m  ({len(segments)} segments)")
    fig.text(0.5, 0.01, info, ha='center', va='bottom', fontsize=9,
             color='#8b949e', family='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22', edgecolor='#30363d'))

    out_path = os.path.join(out_dir, "reward_landscape.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Saved: {out_path}")

    # ── Also generate comparison across deviation_ratio values ──
    fig2, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor='#0d1117')
    fig2.suptitle("Reward Cross-Section Comparison Across Coefficient Values",
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for ax_i, ratio in enumerate(ratios):
        ax = axes.flat[ax_i]
        ax.set_facecolor('#161b22')
        p2 = dict(params)
        p2["spline_deviation_ratio"] = ratio

        _, rad2, pg2, dg2, tg2, _, _ = compute_reward_landscape(
            segments, normals, p2, arc_steps, radial_steps, max_radial)

        # Plot slices at several arc positions
        for k, sa in enumerate(slice_arcs):
            idx = np.argmin(np.abs(arc_pos - sa))
            c = colors_list[k]
            ax.plot(rad2, tg2[idx, :], '-', color=c, lw=1.5, alpha=0.8)

        ax.axvline(gate_w/2, color='cyan', ls='--', lw=1, alpha=0.6)
        ax.axvline(-gate_w/2, color='cyan', ls='--', lw=1, alpha=0.6)
        ax.axhline(0, color='#555', ls='-', lw=0.5)
        ax.set_title(f"deviation_ratio = {ratio}", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Radial offset (m)", color='#8b949e', fontsize=9)
        ax.set_ylabel("s(p)", color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e')
        ax.grid(True, alpha=0.12, color='#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path2 = os.path.join(out_dir, "reward_ratio_comparison.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"✓ Saved: {out_path2}")

    # ── Deviation penalty coefficient comparison ──
    fig3, axes3 = plt.subplots(2, 3, figsize=(20, 10), facecolor='#0d1117')
    fig3.suptitle("Reward Cross-Section Comparison Across Deviation Penalty Coefficients",
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    coeffs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    for ax_i, coeff in enumerate(coeffs):
        ax = axes3.flat[ax_i]
        ax.set_facecolor('#161b22')
        p3 = dict(params)
        p3["deviation_penalty_coeff"] = coeff

        _, rad3, pg3, dg3, tg3, _, _ = compute_reward_landscape(
            segments, normals, p3, arc_steps, radial_steps, max_radial)

        mid = arc_steps // 2
        ax.plot(rad3, pg3[mid, :], '-', color='#3fb950', lw=2, label='spline_progress')
        ax.plot(rad3, dg3[mid, :], '-', color='#f85149', lw=2, label='deviation_penalty')
        ax.plot(rad3, tg3[mid, :], '-', color='#d2a8ff', lw=2.5, label='total')
        ax.axvline(gate_w/2, color='cyan', ls='--', lw=1, alpha=0.6)
        ax.axvline(-gate_w/2, color='cyan', ls='--', lw=1, alpha=0.6)
        ax.axhline(0, color='#555', ls='-', lw=0.5)
        ax.set_title(f"dev_penalty_coeff = {coeff}", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Radial offset (m)", color='#8b949e', fontsize=9)
        ax.set_ylabel("Reward", color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e')
        ax.grid(True, alpha=0.12, color='#30363d')
        if ax_i == 0:
            ax.legend(fontsize=7, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path3 = os.path.join(out_dir, "reward_devcoeff_comparison.png")
    fig3.savefig(out_path3, dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"✓ Saved: {out_path3}")

    return arc_pos, rad_off

def main():
    start_pos, gates_pos, gate_normals, params = load_config()
    segments, total_arc = build_spline(start_pos, gates_pos, gate_normals)
    normals = compute_normals_at_samples(segments)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    render_static(segments, normals, params, start_pos, gates_pos, total_arc, out_dir)

    print(f"\n📊 Current config values:")
    print(f"   spline_deviation_ratio = {params['spline_deviation_ratio']}")
    print(f"   deviation_penalty_coeff = {params['deviation_penalty_coeff']}")
    print(f"   gate_width = {params['gate_width']}")
    print(f"   total arc length = {total_arc:.1f}m across {len(segments)} segments")

if __name__ == "__main__":
    main()
