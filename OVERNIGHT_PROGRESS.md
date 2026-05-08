# Overnight Progress — 2026-05-03 → 2026-05-04

This file is updated live by Claude every loop iteration. Read top-to-bottom for chronological events.

**Goal (this run):** circle_8 + figure_8 deterministic full-lap (≥8/8 gates), then PPO sweep, then final policy push.

(Previous overnight log archived inline below at the bottom.)

---

## Status snapshot

- **Status: FINAL — overnight run wrapping up.**
- **No active training.**
- **Last update:** 2026-05-04 01:01 UTC

## Per-rung table (live, v2 chain)

| # | Track | Path | Steps | Det eval gates | Det eval R | Status | Viz |
|---|---|---|---|---|---|---|---|
| A | straight_2 | curriculum_v2/A_straight_2 | (prior) | 2/2 | — | ✅ | — |
| B | straight_3 | curriculum_v2/B_straight_3 | (prior) | 3/3 | — | ✅ | — |
| C | bent_3_15 | curriculum_v2/C_bent_3_15 | (prior) | 3/3 | — | ✅ | — |
| D | bent_3_30 | curriculum_v2/D_bent_3_30 | 25M added (68.85M total) | 1/3 | 7.89 | ❌ det fail (std too high) | — |
| E | circle_8 | curriculum_v2/E_circle_8 | 50M (R=121 stoch best) | 2/8 | 24.82 | ⚠️ std=1.31 too high | — |
| E2 | circle_8 v2 | curriculum_v2/E_circle_8_v2 | +30M ent=0.0003 (R=479 stoch) | **6/8** | 83.58 | ✅ best so far | — |
| E3 | circle_8 v3 | curriculum_v2/E_circle_8_v3 | +20M ent=0.0001 (R=531 stoch best) | best=5/8, **final=7/8** | 93.69 | ✅ best yet (use FINAL) | — |
| E4 | circle_8 v4 | curriculum_v2/E_circle_8_v4 | +20M ent=0.00005 from v3 final (R=554 stoch) | **best=10/8 ✅** final=9/8 | 131.60 | ✅ **SOLVED** 20/20 | viz/E_circle_8_v4/ |
| F | figure_8 | curriculum_v2/F_figure_8 | +50M ent=0.003 (R=10.5 stoch) | best=1/8 final=1/8 | 10.29 | ❌ stuck on G0 | — |
| F2 | figure_8 v2 | curriculum_v2/F_figure_8_v2 | +30M ent=0.01 (R=10.6 stoch) | best=1/8 final=1/8 | 11.11 | ❌ same plateau | — |
| F3 | figure_8 v3 | curriculum_v2/F_figure_8_v3 | +30M ent=0.005 from straight_3 (R=36 stoch) | best=3/8 final=3/8 | 29.35 | ✅ broke plateau | — |
| F4 | figure_8 v4 | curriculum_v2/F_figure_8_v4 | +30M ent=0.005 cont (R=46 stoch) | best=5/8 final=5/8 | 44.45 | ✅ +2 gates | — |
| F5 | figure_8 v5 | curriculum_v2/F_figure_8_v5 | +30M ent=0.001 (R=47.79 stoch) | best=5/8 final=5/8 | 47.44 | ⚠️ no new gates | — |
| F6 | figure_8 v6 | curriculum_v2/F_figure_8_v6 | +20M ent=0.0003 (R=48.5 stoch) | best=5/8 final=5/8 | 49.04 | ⚠️ same plateau | viz/F_figure_8_v6/ |
| Sweep | sweep_figure_8 | 6 trials × 5M ent×lr | done; ALL = 5/8 | 5/8 | 38-46 | ⚠️ HP not the bottleneck | — |
| F7 | figure_8 v7 final | curriculum_v2/F_figure_8_v7_final | +50M lr=1e-4 ent=0.001 (R=49.94 stoch) | best=5/8 final=3/8 | 49.27 | ⚠️ same plateau | — |
| F8 | figure_8 v8 gate_bonus | curriculum_v2/F_figure_8_v8_gatebonus | +30M gate_bonus=5 (R=80 stoch) | best=3/8 final=3/8 | 41.82 | ⚠️ regressed (early best) | — |
| F8r | figure_8 v8 resume | curriculum_v2/F_figure_8_v8_gatebonus_resume | +17M cont gate_bonus=5 | **best=6/8 ✅** final=6/8 | 80.16 | ✅ **5/8 plateau BROKEN** | viz/F_figure_8_v8_resume/ |
| F9 | figure_8 v9 | curriculum_v2/F_figure_8_v9 | +25M cont (R=80.7 stoch) | best=6/8 final=6/8 | 81.25 | ✅ stable 6/8 | viz/F_figure_8_v9_LONGRUN/ |
| F10 | figure_8 v10 | curriculum_v2/F_figure_8_v10_bonus10 | +30M gate_bonus 5→10 (R=110.9 stoch) | best=6/8 final=6/8 | 111.59 | ⚠️ bigger bonus didn't help | — |
| F11 | figure_8 v11 | curriculum_v2/F_figure_8_v11_explore | +20M ent=0.01 (KILLED) | — | — | killed before ent_bump finished | — |
| F12 | figure_8 v12 distrinit | curriculum_v2/F_figure_8_v12_distrinit | +30M distributed_init=true | best=0/8 final=0/8 | 4.30 | ❌ Phase 2 didn't engage; killed start_pos behavior | — |
| G13 | mini-track left v13 | curriculum_v2/G_figure_8_left_v13 | +15M on 4-gate left lobe only | 2/4 (G4,G5; crashes G5→G6) | 30.79 | ❌ same maneuver wall | — |
| G14 | mini-track left v14 explore | curriculum_v2/G_figure_8_left_v14_explore | +20M ent=0.01 lr=3e-4 | 2/4 same | 31.46 | ❌ exploration didn't help | — |

## Best prior models (carry-forward candidates)

- `saved/curriculum/rung06_circle_8_v4/best_model.zip` — **5/8 gates det, R=60.66, ep_len=203, std=0.208.** Fails by altitude-collapse after G4. Best circle_8 we have entering tonight.
- `saved/curriculum/rung12_figure_8/best_model.zip` — partial training, ep_rew_mean ~8 at last log line.

## Reward / dynamics state (`flightlib/configs/quadrotor_env.yaml`)

- mass=0.73, inertia=[0.003,0.003,0.005], kappa=0.01 (paper)
- act_coeff=0, gate_bonus=0, survival_bonus=0, crash_penalty=0 (paper-strict)
- body_rate_coeff=0.0005, anneal 5M (paper Eq.4)
- gate_width=5
- safety_coeff=0, deviation_penalty_coeff=0
- spline_deviation_ratio=0.5
- distributed_init=false (always start from start_pos)

---

## ⭐ FINAL RESULTS — what to look at in the morning

### Circle_8 — fully solved (10/8 deterministic)

**Best model:** `saved/curriculum_v2/E_circle_8_v4/best_model.zip` (+vec_normalize.pkl)
- Deterministic: 10/8 gates, 20/20, R=131.60, ep_len=392 (~7.8s), avg 20.34 m/s

**Demo GIFs (best to view):**
- `viz/E_circle_8_v4_LONGRUN/best_model_20260503_192149/best_model_ep1_R760_timeout.gif` — **42 gates / 5+ laps / 30s no-crash** (stochastic; 70/200 episodes hit this regime)
- `viz/E_circle_8_v4/best_model_20260503_112329/best_model_ep1_R132_ground_crash.gif` — clean deterministic, 10 gates
- `viz/E_circle_8_v2_LONGRUN/best_model_20260503_192924/best_model_ep1_R756_timeout.gif` — same 42-gate ceiling, different seed

### Figure_8 — partial (6/8 deterministic)

**Best model:** `saved/curriculum_v2/F_figure_8_v8_gatebonus_resume/best_model.zip` (canonical 6/8) or `F_figure_8_v9/best_model.zip` (also 6/8, slightly better R)

**Demo GIFs:**
- `viz/F_figure_8_v8_resume/best_model_20260503_203833/best_model_ep1_R80_ground_crash.gif` — passes G0–G5, crashes during bank to G6
- `viz/F_figure_8_v9_LONGRUN/best_model_20260504_010103/best_model_ep1_R112_ground_crash.gif` — stochastic best, same 6 gates

**Failure mode:** drone reliably reaches G5 then ground-crashes during the sharp left-bank toward G6. The bank requires aggressive coordinated thrust to maintain altitude; the policy learned the right-lobe pattern (CW) but can't execute the curvature reversal at the same level.

### What broke each plateau
- **circle_8 5/8 → 10/8:** entropy compression sweep ent=0.003 → 0.0001 → 0.00005 across v2 → v3 → v4. Key insight: `final_model.zip` can be much better than `best_model.zip` (best is saved on noisy stochastic R, deterministic policy keeps improving past peak stoch).
- **figure_8 5/8 → 6/8:** introduced `gate_bonus=5` (curriculum aid, NOT paper-strict — see FIXES_ATTEMPTED Bug O), then continued training v8 → v8_resume.

### What did NOT break the figure_8 6/8 plateau (everything tried)
- More iterations (v9: +25M same recipe → 6/8)
- Bigger gate_bonus (v10: 5→10 → 6/8)
- Higher entropy (v11: ent=0.01 → killed before finishing)
- Distributed init Phase 1 (v12: 30M → catastrophic 0/8 from start_pos because Phase 2 never engaged with random hover spawns)
- Mini-track on just the failing left lobe (v13/v14: 2/4, same G5→G6 wall)
- PPO HP sweep (6 trials × 5M earlier: all converged at 5/8)

**Conclusion:** The G5→G6 maneuver appears to be a policy-representation limit, not a data-exposure or HP problem. Likely needs either a larger network (512×512), a different action space (CTBR instead of per-motor thrusts), or BC warm-start from a scripted figure_8 trajectory.

### Reward / dynamics state at end of session
- Same as paper-strict EXCEPT `gate_bonus: 10.0` (was 0). All other reward shaping zero.
- `distributed_init: false` (reverted after v12).

---

## Plan (this run, condensed) — historical

1. Wait for D_bent_3_30 → eval → record.
2. Circle_8 50M from D_bent_3_30 (curriculum_v2 chain). Save E_circle_8.
3. If eval <8/8: continue from best 5/8 circle_8 v4 (alt warm-start). Iterate ent_coef sweep + altitude shaping if needed.
4. Visualize circle_8 best.
5. Figure_8 50M from circle_8 best. Iterate. Visualize.
6. PPO HP sweep (8 conditions × ~3M each from circle_8 best).
7. Final push with best HP config.

Full plan in [OVERNIGHT_INSTRUCTIONS.md](OVERNIGHT_INSTRUCTIONS.md).

---

## Event log

### 2026-05-03 06:18 UTC — handoff
- bent_3_30 (D) fine-tune at 52.2M total steps, ep_rew_mean ≈ 22, ep_len ≈ 265, std=1.39, val_loss=0.21, explained_variance=0.993. About 1 iteration from completion (target = 25M added on top of loaded ~27.3M).
- Best model saved at `saved/curriculum_v2/D_bent_3_30/best_model.zip` (R=22.04).
- OVERNIGHT_INSTRUCTIONS.md written. Dynamic loop set up to check every 5–20 min.

### 2026-05-03 06:48 UTC — D_bent_3_30 finished, eval = 1/3 ❌
- Final: 25M steps added (ended at 68.85M cumulative). Training peaked at ep_rew_mean=163 / ep_len=1500 (full timeout) but std grew to 1.73.
- **Deterministic eval: 1/3 gates, R=7.89, ep_len=89.** Stochastic/deterministic gap is huge (high std).
- Decision: skip another bent_3_30 refinement, proceed directly to circle_8 with ent_coef=0.003 (the value that compressed std on prior overnight rungs). The circle_8 fine-tune will both compress std and learn the new track.

### 2026-05-03 06:49 UTC — Circle_8 launched (PID 39177)
- Track applied: circle_8 (8 gates, 18m radius, z=3, 102.8m total spline arc, distributed_init=false).
- Loaded from D_bent_3_30 best_model + vecnorm.
- Args: `--total-timesteps 50000000 --ent-coef 0.003`. cumulative will be ~118.85M.
- Save: `saved/curriculum_v2/E_circle_8/`. TB: `saved/tb_logs/curriculum_v2_E`. Log: `logs/curriculum_v2_E_circle_8.log`.
- Initial: ep_rew_mean=-1.73, ep_len=70, std=1.73 (from bent_3_30).
- Expected runtime ~83 min @ 10K fps.

### 2026-05-03 08:13 UTC — Circle_8 v1 done; eval = 2/8 ❌ → launch v2 std-compress
- Training reward climbed monotonically: -1.7 → 49 → 121 over 50M. ep_len 70 → 448 (full-episode flying).
- **Deterministic eval: 2/8 gates, R=24.82, ep_len=150.** Same stochastic/det gap pattern: std=1.31 still too high.
- Stochastic R=121 (>1 lap) means the *underlying* policy can do the circle; just needs std compression.
- Launched E_circle_8_v2: ent_coef=0.0003 (10× lower), 30M from E_circle_8 best. PID 42978, ~50 min.
- Tactical note: prior overnight's best (rung06_v4) was R=60.66 / 5/8 with std=0.208. This v2 already starts at higher reward (R=112 stoch) so if std compresses similarly, expect to clear ≥5/8.

### 2026-05-03 09:05 UTC — Circle_8 v2 done, eval = **6/8 ✅** (beats prior best 5/8)
- Stochastic ep_rew_mean=479 / ep_len=1300 (full 30s timeout, ~4.6 laps/episode). Std 1.31→1.06.
- **Deterministic eval: 6/8 gates, R=83.58, ep_len=243** — beats yesterday's rung06_v4 best (5/8).
- All 20 episodes hit gates=6 exactly (zero variance). Failure mode is at gate 7.
- Launched v3: ent_coef=0.0001 (3× lower), 20M from v2 best. PID 45344. Goal: push 6/8 → 8/8 with further std compression. Risk: yesterday's v6_commit at 0.0001 regressed; if v3 hurts, v2 stays the canonical best.

### 2026-05-03 10:32 UTC — Circle_8 v3 done; FINAL gets 7/8 ✅; v4 launched
- v3 best_model.zip evaluated 5/8 (regressed) — best save was triggered early at peak stoch R=531 then later training kept improving without re-saving.
- **v3 final_model.zip evaluated 7/8 gates, R=93.69, ep_len=299** — new best deterministic policy.
- All 20 episodes hit 7/7 exactly. Failure mode: G7 (closing the loop), not G4 anymore.
- Insight: in this env, `final_model` can be substantially better than `best_model` because the saver only fires on stochastic rolling reward. For tonight, eval BOTH best and final on every run.
- Launched v4: ent=0.00005 (half of v3), 20M from v3 FINAL. PID 47698. Goal: 7/8 → 8/8.

### 2026-05-03 11:23 UTC — 🎉 CIRCLE_8 SOLVED with v4
- v4 (ent=0.00005, 20M from v3 final): stoch best R=554, std compressed 0.889 → 0.730.
- **best_model: 10 gates passed, 20/20 episodes, R=131.60, ep_len=392, mean_speed=20.34 m/s.** Drone does >1 full lap.
- final_model: 9/8, 20/20, R=107.38.
- Viz rendered: `viz/E_circle_8_v4/best_model_20260503_112329/` — 3D trajectories PNG, topdown PNG, best/worst episode detail PNGs, 3 animated GIFs, raw trajectory npz.
- All 20 deterministic episodes hit gates=10 / R=131.60 (zero variance — fully deterministic policy).

### 2026-05-03 11:24 UTC — Figure_8 launched (PID 49543)
- Track applied: figure_8 (8 gates, 6m radius lobes, curvature reversal at origin, z=3, start (0,-4,3) facing +y).
- Loaded from circle_8 v4 best (the 10/8 winner). vec_norm carried over.
- Args: `--total-timesteps 50000000 --ent-coef 0.003`. cumulative will be ~218M.
- Save: `saved/curriculum_v2/F_figure_8/`. Initial ep_rew_mean=1.30 (curvature reversal is unfamiliar — circle_8 policy doesn't know how to switch direction).
- Expect ~80 min (50M @ 10K fps).

### 2026-05-03 12:44 UTC — Figure_8 v1 done; eval = 1/8 ❌; v2 launched with ent bump
- Figure_8 v1 (50M ent=0.003 from circle_8 v4 best): stoch R plateaued at 10.46 with ep_len=77 (1.5s episodes), std compressed 0.73→0.494.
- **Det eval: 1/8 gates** (best AND final). Drone passes G0 then crashes. Curvature reversal at origin is unfamiliar — circle_8 prior is CCW-biased, figure_8 needs CW first lobe then CCW.
- Launched v2: ent=0.01 (3.3× higher) to break out of the local "turn left only" basin. 30M from F_figure_8 best. PID 52323. ~50 min.
- If v2 doesn't move past 1/8: try v3 with ent=0.02 OR fresh-start figure_8 from straight_3 (no circle_8 bias).

### 2026-05-03 13:36 UTC — Figure_8 v2 done; same 1/8 plateau; v3 launched from straight_3
- v2 (ent=0.01, 30M from F_figure_8 best): stoch R 10.46 → 10.63 (almost no improvement). Det eval still 1/8 for both best and final.
- ent_bump from 0.003 → 0.01 didn't break the plateau. The drone's policy is heavily locked into the circle_8 CCW prior.
- Launched v3: fresh-start from `B_straight_3/best_model.zip` (the curriculum_v2 endpoint with no directional curve bias). ent=0.005, 30M. PID 54110.
- If v3 doesn't crack figure_8 in 30M: accept and move to PPO sweep on circle_8 (the user explicitly said "try" for figure_8, "must" for circle_8 ✓).

### 2026-05-03 17:42 UTC — Sweep results + final push launched
- Sweep on figure_8 (6 trials, 5M each from F_figure_8_v4): all 6 converged to 5/8 deterministic, R≈42-46. HP knobs (lr 1e-4 / 3e-4 / 5e-4 × ent 0.0001 / 0.005 / 0.02 / 0.05) cannot break the gate-6 plateau.
- Conclusion: bottleneck is structural — drone learns right-lobe well but cannot execute the curvature reversal at G6 (crossover from right lobe back to left lobe).
- **Final push (v7)** launched: 50M, lr=1e-4 (slowest, preserves prior), ent=0.001 (deeper compression), from F_figure_8_v6 best (5/8 with std=0.261). PID 63886, ~80 min.
- Reasoning: long, slow refinement is the only HP path left; if it doesn't work, the next escalation is reward shaping (gate_bonus reintroduction would break paper-strict but would create a step-function signal at gate boundaries).

### 2026-05-03 06:53 UTC — killed orphan D_bent_3_30_v2_refine
- A stale shell (PID 39308) from a prior session auto-launched a SECOND training (PID 39312) with `D_bent_3_30_v2_refine` (15M, ent=0.005, lr=1e-4). Killed both before they could compete with circle_8 for CPU. Single-training invariant restored.

---

## PPO sweep results (Phase 6)

| ID | Config | Path | Det gates | R | Notes |
|---|---|---|---|---|---|
| _none yet_ | | | | | |

## Issues / decisions log

_(empty so far)_

---

## Archive: previous overnight (2026-05-02 → 2026-05-03)

Prior session's circle_8 result: best at `saved/curriculum/rung06_circle_8_v4/`, **5/8 det, R=60.66, ep_len=203, std=0.208**. Failure mode: altitude collapse just after G4 (drone drops z 3 → 0). Tried ent_coef = 0.001 / 0.0003 / 0.05 / 0.0001 — all converged at 5/8 after the v3 breakthrough. Started figure_8 from circle_8 v4 at 16:35 UTC, training reached ~200M total steps but ep_rew_mean only ~8. Recipe-confirmed for bent rungs: lower ent + more iterations.
