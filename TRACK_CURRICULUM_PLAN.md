# Track Curriculum Plan

**Goal:** Get the drone visibly flying through gates by end of day, AND identify the exact track-complexity step at which the policy breaks.

**Strategy:** Build a ladder of tracks where each rung adds **exactly one** new geometric difficulty. Train sequentially, fine-tuning the policy from the previous stage. The most impressive demo is the deepest rung the drone can complete; the science is the rung where it fails.

---

## 0. Pre-flight: 3D track validation tool

Before training on any track, render it in 3D and visually confirm:
- Gate positions are where you expect.
- Gate normals (the fly-through direction) point along the racing line, not perpendicular to it.
- The cubic Hermite spline through `[start_pos, G0, G1, ...]` looks smooth and doesn't loop or kink.
- Vertical scale is sane (gates aren't all stacked at z=0 from a YAML typo).

### Tool

Reuse / extend [flightrl/training/plot_track_layout.py](flightrl/training/plot_track_layout.py) — it already draws gates as 3D rectangles with normal arrows and renders a spline. Wrap it so it accepts a track YAML directly and dumps a PNG to `flightrl/training/track_previews/<track>.png`.

```bash
python plot_track_layout.py --track straight_3 --out track_previews/straight_3.png
```

**Success gate:** Open every PNG before that track enters the curriculum. If the spline tangent at any gate is more than ~30° off the gate's normal, the track is unlearnable — fix the rotations before training.

---

## 1. The ladder

Each rung adds **one** complication on top of the previous. Time budget: ~45 min/rung (15 min build/preview + 25 min training + 5 min eval). With ~12 useful hours, plan for the curriculum to comfortably reach rung 8–10.

| # | Track name | New complication | Gates | Status |
|---|---|---|---|---|
| 0 | `single_1` | Fly to a single gate, 8 m straight | 1 | **build** |
| 1 | `straight_2` | Continue past one gate to the next | 2 | **build** |
| 2 | `straight_3` | Sustained straight-line flight | 3 | exists |
| 3 | `bent_3_15` | Add gentle 15° bend per gate | 3 | **build** |
| 4 | `bent_3_30` | Sharpen to 30° per gate | 3 | **build** |
| 5 | `bent_3_45` | Sharpen to 45° per gate | 3 | **build** |
| 6 | `circle_8` | Continuous (sustained) turning, gentle | 8 | **build** |
| 7 | `circle_6` | Tighter sustained turning (60°/gate) | 6 | exists |
| 8 | `circle_4` | Sharp sustained turning (90°/gate) | 4 | exists |
| 9 | `oval_8` | Varying curvature within a lap | 8 | exists |
| 10 | `vertical_3` | 3-gate straight with 3 m altitude rise | 3 | **build** |
| 11 | `helix_8` | Sustained turning + sustained climb | 8 | **build** |
| 12 | `figure_8` | Direction reversal mid-track | 8 | **build** |
| 13 | `alphapilot` | Full original IROS track | 12 | exists |

**Why this ordering:**
- 0–2 isolate forward flight + gate-pass continuation (no turning).
- 3–5 isolate **transient** turning at increasing severity.
- 6–8 isolate **sustained** turning (continuous yaw rate) at increasing severity. `circle_8` before `circle_6` because larger N = gentler per-gate turn.
- 9 introduces curvature variation within an episode.
- 10–11 introduce the vertical axis.
- 12 introduces direction reversal (gate normals flip mid-track).
- 13 is the full thing — only attempted if 11/12 succeed.

**What "build" means** for each new track: write `flightlib/configs/tracks/<name>.yaml` matching the schema of `straight_3.yaml` (`description`, `start_pos`, `gates: [{position, rotation: [yaw, pitch, roll]}, ...]`). Gate yaw should equal the spline tangent at that gate (atan2(dy, dx) for the segment entering the gate). For the bent_3 family, lay out three collinear gates and rotate each by the per-gate bend so the policy must yaw incrementally.

---

## 2. Per-phase training workflow

Run this exact loop for each rung:

```
1. Build / edit the track YAML.
2. Render the 3D preview PNG. Eyeball it. Fix and re-preview if anything looks off.
3. python track_loader.py apply <track>
4. (First rung only) train from scratch.
   (Later rungs)    --load-model from previous rung's best_model.zip
                    --load-vecnorm from previous rung's vec_normalize.pkl
5. Train for the budgeted step count (see § 3) with TensorBoard open.
6. Run deterministic eval from start_pos (set_random_reset(False)) — N=20 episodes.
7. Record gates_passed, lap_completed, ep_len_mean, ep_rew_mean.
8. If success criteria met → save best_model.zip + vec_normalize.pkl to
   saved/curriculum/<rung>/, log the result, advance.
   If not met after 2× budget → log as the failure rung and stop.
9. Render an eval rollout GIF for the demo gallery (visualize_policy.py --animate).
```

### Transfer details

- Use the **same** PPO hyperparameters across rungs (lr=3e-4, ent_coef=0.01, etc.).
- Carry forward both the policy weights *and* the VecNormalize running stats. Don't reset VecNormalize between rungs — let it adapt online to the new obs distribution. (If a rung's eval is wildly worse than its training reward suggests, suspect VecNormalize drift and try a fresh stats reset on a re-run.)
- Don't freeze any layers. Continued PPO fine-tuning is simpler than freeze-and-thaw and works fine when the obs schema is identical (it is — same 26-dim spec across all tracks).

---

## 3. Step budgets and success criteria

Step budget per rung, based on track length and difficulty. Stop early if criteria are hit; extend up to 2× if you're seeing a clear upward trend on `ep_rew_mean` and `gates_passed`.

| # | Track | Budget (steps) | Success criteria (deterministic eval, 20 episodes) |
|---|---|---|---|
| 0 | `single_1` | 1 M | mean `gates_passed` ≥ 1.0 in ≥ 18/20 |
| 1 | `straight_2` | 1.5 M | mean `gates_passed` ≥ 2.0 in ≥ 18/20 |
| 2 | `straight_3` | 2 M | full lap in ≥ 18/20 |
| 3 | `bent_3_15` | 2 M | full lap in ≥ 18/20 |
| 4 | `bent_3_30` | 2 M | full lap in ≥ 16/20 |
| 5 | `bent_3_45` | 2.5 M | full lap in ≥ 16/20 |
| 6 | `circle_8` | 3 M | full lap in ≥ 16/20 |
| 7 | `circle_6` | 3 M | full lap in ≥ 14/20 |
| 8 | `circle_4` | 4 M | full lap in ≥ 12/20 |
| 9 | `oval_8` | 4 M | full lap in ≥ 12/20 |
| 10 | `vertical_3` | 2 M | full lap in ≥ 16/20 |
| 11 | `helix_8` | 4 M | full lap in ≥ 12/20 |
| 12 | `figure_8` | 5 M | full lap in ≥ 10/20 |
| 13 | `alphapilot` | 10 M | mean `gates_passed` ≥ 6 (i.e. half the track) |

**Hard criterion** for advancing: `eval_lap_completed` true rate above the per-rung threshold. **Soft signal**: `ep_len_mean` should approach the per-track timeout (no more sub-100-step crashes) — this was the symptom that killed the AlphaPilot sweep.

**Failure rung** is the first one where the soft signal stays flat (no increase in ep length or gates passed) for the full extended budget. That's the science result: "the policy learns through complication X but fails at complication Y."

---

## 4. Logging

For each rung, append to `flightrl/training/curriculum_log.md`:

```
### Rung N — <track name>
- Loaded from: <previous rung path or "scratch">
- Steps trained: X
- Eval: gates_passed mean=A.A, lap_completed=B/20, ep_len_mean=C
- Outcome: ✅ advanced | ❌ failed | ⏸ partial
- Failure mode (if ❌): one sentence on what the policy was doing wrong
- Preview PNG: track_previews/<track>.png
- Demo GIF: demos/<track>.gif
```

This becomes the writeup for the report.

---

## 5. End-of-day deliverables

1. `track_previews/*.png` — 3D visualization of every rung in the ladder.
2. `saved/curriculum/<rung>/` — trained policy weights for every rung that succeeded.
3. `demos/<track>.gif` — rollout animation for at least the deepest succeeded rung (this is the demo).
4. `curriculum_log.md` — table of results, including the failure rung and its failure mode.
5. Updated `FIXES_ATTEMPTED.md` if any reward / dynamics tweak was needed mid-curriculum to clear a rung.

---

## 6. Risk mitigations

- **Rung 0 fails (drone can't even fly to a single gate):** This is a *dynamics or reward* problem, not a track problem. Drop max thrust 30%, double `gate_width` on the test track, and add a small flat survival bonus for the first rung only. Document the deviation; restore for later rungs.
- **VecNormalize stats blow up on transfer:** Save and reload them; if values look extreme on the next rung, reset stats and let them re-fit for ~50K steps with a frozen policy (or just accept the early degradation — it usually self-corrects within an update or two).
- **Time pressure forces a skip:** The minimum impressive demo is **rung 2** (full straight_3 lap). Anything beyond that — even a circle — is a solid result. Don't burn the day chasing rung 13.
