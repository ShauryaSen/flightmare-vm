# Fixes Attempted — Drone Racing RL

Append-only log of debugging hypotheses tried for the "drone gets stuck before first gate" problem (and related issues). Maintained via the [fix-tracker](.claude/skills/fix-tracker/SKILL.md) skill.

**Read this BEFORE proposing a new fix.** Failed attempts are the most valuable entries — they tell you what NOT to try again.

---

## 2026-04-30 — Bug A: Remove progress reward double-clamping

**Hypothesis:** The implementation clamps the projected progress in two places: `segment_progress()` clamps the projection itself to `[0, seg_len+0.5]`, AND `step()` clamps the per-step delta to `±delta_max=0.5`. This caps forward progress at 25 m/s (so longer segments yield zero gradient) and zeroes the signal entirely once the drone overshoots the gate plane or drifts behind the segment start. Paper Eq. 2 uses raw unclamped projection.

**Files changed:**
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:502-512` — `segment_progress()` returns raw projection, no clamp
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:286-291` — removed `±delta_max` clamp on `s_curr - s_prev_`

**Expected effect:** `ep_rew_mean` curve grows roughly linearly with distance traveled rather than plateauing; drone clears gate 0 within ~1-2 M steps where it currently never does.

**Outcome:** _pending — to be filled in after user tests_

---

## 2026-04-30 — Bug B: Strip non-paper reward terms

**Hypothesis:** The implementation added `gate_bonus` (+5/+10), `survival_bonus` (+0.1/step), `act_coeff·‖a‖²`, and a flat `crash_penalty` (-30) on ground crash / OOB. Paper Eq. 4 has none of these — only `r_p + a·r_s − b·‖ω‖²` plus `r_T` triggered exclusively on gate collision. User has empirically confirmed toggling these does not unstick training, suggesting they obscure the progress gradient rather than help it. Removing them also makes ablations cleaner.

**Files changed:**
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:317-339` — removed `gate_bonus_` add on pass; gate index still advances
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:353-354` — removed `act_coeff` and `survival_bonus` adds
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:362-379` — `isTerminalState()` now returns 0 on ground crash / OOB / timeout; only gate collision yields `r_T`
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp:139-156` — deleted dead members
- `flightlib/configs/quadrotor_env.yaml:78-93` — zeroed/removed corresponding fields

**Expected effect:** No change in qualitative behavior on its own (since the user already confirmed these terms don't help); but combined with Bug A, removes confounders that could mask the progress signal. Reward magnitudes will look much smaller (no +0.1/step floor; no +5/10 spikes).

**Outcome:** _pending — to be filled in after user tests_

---

## 2026-04-30 — Bug C: Reduce gate_width from 3.0 m to 1.2 m

**Hypothesis:** `gate_width: 3` made the pass test `d_p < 1.5 m` cover a 3 m disk around each gate axis, so collisions almost never registered. The drone could "pass" gates while flying far off-axis, learning incorrect geometry. AlphaPilot/Split-S real gates are ~1.2 m square.

**Files changed:**
- `flightlib/configs/quadrotor_env.yaml:88` — `gate_width: 3 → 1.2`

**Expected effect:** Gate collision penalty (`r_T`, paper Eq. 5) actually fires on misses, providing the negative shaping the paper relies on. Early-training crash count will rise sharply (this is intended — those were silently being counted as passes before).

**Outcome:** _pending — to be filled in after user tests_

---

## 2026-04-30 — Bug D: Inertia matrix and kappa to paper values

**Hypothesis:** `QuadrotorDynamics::updateInertiaMarix()` auto-computes `J = m/12·arm²·diag(4.5, 4.5, 7)`, giving `J_xx ≈ 0.0079, J_zz ≈ 0.012` for the 0.73 kg / 0.17 m drone. Paper experiments.tex L44 specifies `J = diag(0.003, 0.003, 0.005)` — **the auto-computed inertia is ~2.5× heavier**. A heavier drone is sluggish: the random initial policy can't accelerate or rotate fast enough to make progress before drifting out of position. Also `kappa = 0.016` vs paper `0.01` (rotor yaw torque, smaller effect).

**Files changed:**
- `flightlib/src/dynamics/quadrotor_dynamics.cpp:181-191` — `updateParams()` reads optional `inertia` array and overrides auto-computed `J_`
- `flightlib/configs/quadrotor_env.yaml:13-15` — added `inertia: [0.003, 0.003, 0.005]`, set `kappa: 0.01`

**Expected effect:** Drone is more agile from step 1 of training. Random-policy episodes should show wider attitude variation and higher angular velocities. Combined with Bug A, this should let the policy actually traverse the 18.6 m first segment before timing out.

**Outcome:** _pending — to be filled in after user tests_

---

## 2026-04-30 — Bug E: Clean up Phase-1 spawn t formula

**Hypothesis:** The line `t = 0.2 + uniform·0.3 + 0.25` had a vestigial `+0.25` constant. With `uniform_dist_ = U(-1, 1)`, this gave `t ∈ [0.15, 0.75]` — never close to the start of the segment. The paper says spawn "around the centers of all path segments," which suggests a symmetric distribution centered on 0.5 with broader coverage.

**Files changed:**
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:107-110` — `t = 0.5 + U(-1,1)·0.3 → [0.2, 0.8]`

**Expected effect:** Marginal — minor improvement in distributed-init coverage. Not load-bearing for the stuck-at-spawn issue. Mostly a cleanup.

**Outcome:** _pending — to be filled in after user tests_
