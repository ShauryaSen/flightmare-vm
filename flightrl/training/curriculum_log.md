
# Track Curriculum — Training Log

Sequential progress through [TRACK_CURRICULUM_PLAN.md](../../TRACK_CURRICULUM_PLAN.md).

**Date:** 2026-05-01 → 2026-05-02
**Total training compute:** ~42M steps across 6 rungs

---

## TL;DR

| Rungs cleared | Failure rung | Hardest demo |
|---|---|---|
| **0, 1, 2, 3, 10** (5 / 13) | **Rung 4 (`bent_3_30`)** plateaued at 2/3 gates after 8M steps. Rung 6 (`circle_8`) confirms the ceiling — drone manages 1/8 gates on a circle. | **Rung 10 — `vertical_3`** (climbing 3-gate full lap, 20/20 deterministic). Tied with Rung 3 (`bent_3_15`) for sharpest learned trajectory. |

**Science finding (per plan §3):** the policy successfully learns through complication "transient bend ≤15° per gate" (rung 3) but fails at "transient bend ≥30° per gate" (rung 4). Sustained-turning tracks (rung 6 onward) inherit the same ceiling — the drone can pass one gate of a circle but cannot maintain the constant-yaw-rate trajectory the rest requires.

---

## Reward shape (paper-faithful, fixed across all rungs)

- `survival_bonus = 0.0`
- `crash_penalty = 0.0`
- `gate_bonus = 0.0`
- `body_rate_coeff = 0.0005` (annealed → 0 over 5M steps, paper L140)
- `spline_deviation_ratio = 0.5`, `deviation_penalty_coeff = 0.0`
- `gate_width = 5` (larger than paper's 1.2 for easier credit assignment, kept consistent across rungs)
- `distributed_init = false` (always reset to `start_pos` for curriculum learning)

## PPO hyperparams

`lr=3e-4`, default `ent_coef=0.01` (occasional bumps to 0.02/0.03 documented per rung), `batch=150K`, `n_epochs=10`. 200 envs × 750 n_steps.

## Transfer rule

Every rung loads the previous rung's `best_model.zip` + `vec_normalize.pkl`. Rung 0 trains from scratch.

---

## Status table

| # | Track | Steps | gates_passed (deterministic eval, 20 eps) | Lap rate | Outcome | Notes |
|---|---|---|---|---|---|---|
| 0 | single_1 | 1.05M (scratch) | 1.00 ± 0.00 | 20/20 | ✅ PASS | ent=0.01, R=-9.18, ep_len=186 |
| 1 | straight_2 | 1.5M (← 0) | 2.00 ± 0.00 | 20/20 | ✅ PASS | ent=0.01, R=-2.89, ep_len=163 |
| 2 | straight_3 | 4M (← 1) | 3.00 ± 0.00 | 20/20 | ✅ PASS | first 2M plateaued at 2/3; +ent=0.02 + 2M extension broke through. R=15.56, ep_len=143 |
| 3 | bent_3_15 | 12M (← 2) | 3.00 ± 0.00 | 20/20 | ✅ PASS | needed 4 sub-runs (v1=ent=0.02, v2=0.02 ext, v3=0.015 ext, v4=0.01 ext); **lowering ent_coef back to default cracked it.** R=16.30, ep_len=118 |
| 4 | bent_3_30 | 8M (← 3) | 2.00 ± 0.00 | 0/20 ≥3 | ⚠️ PARTIAL — failure rung | Plateaus at 2/3 across 8M (2× budget). 30°-per-gate exceeds the trained policy's bend capability with paper-strict reward + 0.73 kg / paper-inertia drone. R=16.84. |
| 5 | bent_3_45 | _not attempted_ | — | — | ⏸ skipped | After rung 4 hit the ceiling, harder bent track skipped; would not pass given the 30° pattern. |
| 6 | circle_8 | 8M (← 3, then ext) | 1.00 ± 0.00 | 0/20 ≥8 | ❌ FAIL | First 4M: drone gets ~12 R but only 1 gate; ep_len=125. Second 4M extension: training reward briefly climbed to 14 then settled, eval still 1/8. Confirms sustained turning is the load-bearing missing capability. R=11.46. |
| 7 | circle_6 | _not attempted_ | — | — | ⏸ skipped | 60°/gate sharper than circle_8; would not pass. |
| 8 | circle_4 | _not attempted_ | — | — | ⏸ skipped | 90°/gate; would not pass. |
| 9 | oval_8 | _not attempted_ | — | — | ⏸ skipped | Inherits circle ceiling. |
| 10 | vertical_3 | 2M (← 3) | 3.00 ± 0.00 | 20/20 | ✅ PASS | 3-gate climb (1m altitude/gate, 7° pitch). Confirms hypothesis: altitude-axis change alone is easy; the rung-4 ceiling really is about *yaw*, not 3D motion in general. R=16.84, ep_len=119. |
| 11 | helix_8 | _not attempted_ | — | — | ⏸ skipped | Inherits circle ceiling. |
| 12 | figure_8 | _not attempted_ | — | — | ⏸ skipped | Inherits circle ceiling. |

---

## Rung-by-rung notes

### Rung 0 — `single_1` ✅
- Loaded from: scratch
- Steps: 1.05M (1M budget, finished at iteration 7)
- Eval: 1.00 ± 0.00 gates/ep, lap_completed = 20/20, ep_len_mean = 186 steps
- Outcome: ✅ advanced
- Reward trajectory: -26.4 → -17.7 → -12.1 → -8.7 → -5.8 → -3.3 (training rolling mean)
- Demo: not yet rendered; trajectories for visualize_policy.py work from `saved/curriculum/rung00_single_1/best_model.zip`

### Rung 1 — `straight_2` ✅
- Loaded from: rung 0
- Steps: 1.5M (budget hit)
- Eval: 2.00 ± 0.00 gates/ep, lap=20/20, ep_len=163
- Outcome: ✅ advanced
- Reward: -4.7 → -1.9 → -1.4 → -0.5 (final)

### Rung 2 — `straight_3` ✅ (after extension)
- Loaded from: rung 1
- v1 (2M, ent=0.01): plateaued at 2/3 gates eval, R=18.2 ep_len=135
- v2 (+2M, ent=0.02): broke through, full 3/3 lap (20/20). Final R=15.56 ep_len=143
- Lesson: at the straight→multi-gate transition, drone tends to overshoot G2 and miss G3 — bumping ent_coef forces it to explore committing earlier.

### Rung 3 — `bent_3_15` ✅ (after 4 sub-runs)
- Loaded from: rung 2
- v1 (2M, ent=0.02): 2/3 plateau, R=16.7
- v2 (+2M, ent=0.03): still 2/3, R=17.7. The std of the policy was climbing (1.05 → 1.17), suggesting ent_coef was preventing commitment.
- v3 (+4M, ent=0.015): still 2/3
- v4 (+4M, ent=0.01): **broke through** — 3/3 in 20/20. R=16.30 ep_len=118
- Total: 12M steps. Insight: too-high ent_coef on bent tracks **prevents** the policy from committing to a single bent trajectory. Restoring ent_coef=0.01 was the unlock.

### Rung 4 — `bent_3_30` ⚠️ (failure rung)
- Loaded from: rung 3 (which had just learned 15°/gate)
- v1 (4M, ent=0.01): 2/3 plateau eval, R=17.04
- v2 (+4M, ent=0.01): still 2/3, R=16.84
- Total 8M = 2× budget per plan, no movement off the 2/3 plateau.
- Failure mechanism: G2 of bent_3_30 is at heading 60° from start. The policy carries momentum from G0 (heading 0°) and G1 (heading 30°) but cannot bank hard enough to redirect to G2 in time. Drone overshoots laterally past G2, episode terminates around step 125 (2.5s) by drift OOB or ground crash on the recovery.
- Per plan §3, this is the documented failure rung — "the policy learns through complication X (15°/gate) but fails at complication Y (30°/gate)".

### Rung 6 — `circle_8` ❌
- Loaded from: rung 3 (skipping bent_3_30/45 ceiling)
- v1 (4M, ent=0.01): training reward swung -2.6 → -2.5 → -2.4 → -1.6 → +1.5 → +6.3 → +10.2 → +12.5 → +14.1. Looked promising. Eval: 1/8 gates, R=12.69, ep_len=125.
- v2 (+4M, ent=0.01): training reward stayed in the +14 range, value_loss dropped further. Eval: still 1/8, R=11.46, ep_len=140.
- Failure mechanism: drone learns a "fly toward G0 fast" strategy that earns the bulk of in-rollout reward (each spline-progress chunk is highly rewarded) but commits to too high a forward velocity to bank into G1's offset. After passing G0, drone overshoots the circle's tangent and drifts out.
- The training/eval reward gap (+14 stochastic vs +11 deterministic) suggests the stochastic policy occasionally banks correctly but the deterministic mean does not.

---

## Saved artifacts

```
flightrl/training/saved/curriculum/
├── rung00_single_1/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/,tb_v4/}
├── rung01_straight_2/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/,tb/}
├── rung02_straight_3/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/}
├── rung03_bent_3_15/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/}
├── rung04_bent_3_30/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/}    ← partial
└── rung06_circle_8/{best_model.zip,vec_normalize.pkl,final_model.zip,checkpoints/}     ← failed
```

Track previews (3D PNGs, generated before training):
```
flightrl/training/track_previews/{single_1,straight_2,straight_3,bent_3_15,bent_3_30,bent_3_45,
  circle_8,circle_6,circle_4,oval_8,vertical_3,helix_8,figure_8,alphapilot}.png
```

Eval logs alongside each checkpoint dir as `eval.log`.
TB logs in `saved/tb_logs/curriculum/` for inspection: `tensorboard --logdir saved/tb_logs/curriculum`.

---

## Why training stalled at the bend ceiling

Three plausible causes, each consistent with the plateau pattern:

1. **Action space:** per-motor thrusts give the policy fine control but make banking a 4-D coordination problem; the paper used the same action space and reports needing 25M+ steps for full track. Our 4M budgets per rung may simply be insufficient on aggressive bends.
2. **Drone dynamics:** mass = 0.73 kg, paper inertia [0.003, 0.003, 0.005], gate_width = 5m. The agile dynamics should support 30°+ banking, but the wide gate makes "skim past with shallow turn" a dominant strategy on rung 3 that doesn't transfer.
3. **Reward shape:** paper-strict (no crash_penalty, no survival_bonus) means terminating the episode after a missed turn costs only the lost forward progress — there's no signal saying "cleared 3 gates > overshot at gate 3". The policy converges to a local optimum: maximize per-step progress until the gate that breaks the linear strategy.

The user's "increase entropy if face-planting" prescription was effective on **straight tracks** (rung 2 needed it) but **counter-productive on bent tracks** (rung 3 needed entropy *lowered* before it broke through). The lesson — which deviated from the user's input but matched the empirical signal — is documented above.

## Recommended next experiments

1. **Higher-resolution failure-rung characterization** (the science output): retrain rung 4 from scratch at 0/5/10/20/30 M-step checkpoints and plot gates_passed vs steps. Quantify the ceiling.
2. **Smaller `gate_width`** (1.2 m, paper value): forces precision; may eliminate the "skim past at the wrong angle" loophole.
3. **Reintroduce a small `gate_bonus`** (deviates from paper, but explicitly motivated): ~5 reward per gate would create a step function at gate boundaries, forcing the policy to commit to gates rather than pure spline-following.
4. **IL warm-start from a tracker on bent tracks**: trajectory_controller.py can produce demos that bend correctly; BC pretrain + low-LR PPO fine-tune (the workflow that already works on circle_8) could break the rung-4 ceiling.
