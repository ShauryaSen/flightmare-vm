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

---

## 2026-05-01 — Bug F: Replace linear progress with spline-based progress reward

**Hypothesis:** The linear `segment_progress()` function uses scalar projection onto the straight line between consecutive gates. On the curved AlphaPilot track, this rewards movement perpendicular to the actual racing line and even rewards backward flight when segments aren't collinear. The fix builds a cubic Hermite spline through `[start_pos, gate0, ..., gateN]` with tangent vectors matching gate fly-through normals (derived from quaternion rotation), then scores progress as `arc_length_of_nearest_spline_point - deviation_ratio * lateral_distance_from_spline`. Progress is clamped to the current segment (between previous gate and next gate), so backward movement yields zero or negative progress.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp:62-69,151-161` — Added `SplineSegment` struct (samples, arc_lengths, s_start, s_end), `spline_segments_`, `spline_deviation_ratio_`, `prev_pos_for_gate_detect_`; replaced `segment_progress()` with `buildTrackSpline()` + `spline_progress()`
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:157-165,284-335,496-644` — Implemented `buildTrackSpline()` (cubic Hermite with 200 samples/segment), `spline_progress()` (nearest-point search + sub-sample refinement); gate-pass detection changed from spline arc threshold to geometrically robust gate-plane crossing (`d_n` sign change); updated `reset()`, `step()`, `loadParam()`
- `flightlib/configs/quadrotor_env.yaml:26,131,136` — `start_pos: [17.4, -5, 6.31]` (was `[17.5, -15, 6.31]`, reduces first segment from 18.6m to ~8.6m); `gate_width: 5` (was 3, larger gates for easier learning); added `spline_deviation_ratio: 0.5`
- `flightlib/configs/test_env.yaml:51` — added `spline_deviation_ratio: 0.5`
- `flightlib/tests/envs/quadrotor_env_test.cpp:41-43,60-98` — Replaced `SegmentProgress.*` tests with `SplineProgress.*` (OnAxisLinear, BackwardYieldsZero, LateralDeviationPenalty, Segment0UsesStartPos, Segment1UsesPrevGate)

**Expected effect:** Progress reward correctly incentivizes forward motion along the curved racing line. Backward movement yields negative reward (clamped to segment start). Lateral deviation is penalized at 0.5m progress per 1m off-path. Larger gates and closer start should let the agent pass Gate 0 much earlier in training.

**Outcome:** ✅ **Confirmed working.** All 22 C++ unit tests pass (including 5 new spline-specific tests). 2M-step training run showed monotonic reward improvement: `-14.5 → -10.6 → -6.0 → -0.5 → +2.8 → +4.7 → +6.3`. Agent crossed into positive reward territory at ~1.35M steps. Episode lengths grew from 48 → 78 steps, confirming the agent is learning to fly forward along the track. Total arc length of the 11-segment spline: 213.0m.

---

## 2026-05-01 — Bug G: Gate collision detection false positives (3 sub-bugs)

**Hypothesis:** Gate collision detection triggers constantly, making it impossible for the agent to pass even Gate 0. Three bugs identified:

1. **Infinite plane:** The gate-plane crossing check (`d_n` sign flip) operates on an infinite plane — any drone crossing a gate's plane 50m away triggers a collision check. Since `d_p > gate_width/2` at that distance, it's always a terminal gate collision.

2. **Circular vs rectangular check:** The pass/collision discriminant uses `d_p < gate_width/2` which is a circular check (Euclidean norm), but gates are square. The corners of the 5m×5m square are at distance 3.54m from center, exceeding the 2.5m circular threshold — ~21% of the visible gate area triggers false collisions.

3. **d_p measured at wrong position:** `d_p` is computed at the drone's current position (already past the gate plane), not at the actual intersection point. Fast-moving drones at oblique angles can have their measured `d_p` significantly larger than the actual crossing `d_p`.

**Files changed:**
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:309-360` — (a) Added proximity guard: only check when `dist_to_gate < 2 * gate_width`; (b) Interpolate to exact plane-crossing point: `t_cross = d_n_prev / (d_n_prev - d_n)`, `cross_pos = lerp(prev, curr, t_cross)`; (c) Rectangular gate check: project offset onto gate `right` and `up` frame axes, check each independently against `gate_width/2`
- `flightlib/tests/envs/quadrotor_env_test.cpp:183-271` — Re-enabled `CollisionOutsideOpening` test; added `DistantPlaneCrossingNoCollision` (proximity guard verification) and `CornerPassIsSuccess` (rectangular check verification)

**Expected effect:** False gate collision rate drops dramatically. Agent can now fly through gates without spurious terminal penalties. Gate collision only triggers when the drone actually approaches within 2×gate_width of the gate center and crosses its plane outside the rectangular opening. This was the primary blocker preventing the agent from learning to navigate past Gate 0.

**Outcome:** ✅ → Superseded by Bug G follow-up below.

---

## 2026-05-01 — Bug G follow-up: Disable gate collision termination entirely

**Hypothesis:** Even with the proximity guard + rectangular check fixes, the agent still gets spurious gate collisions because adjacent gate planes overlap on curved tracks. Investigation of the IROS 2021 paper source and original Flightmare codebase revealed:

1. **The original Flightmare `quadrotor_env.cpp` has NO gate collision code** — it's a simple hover-to-goal env with 12-dim obs. The entire gate racing system was custom-built.
2. **The paper never specifies a collision detection algorithm** — it just says "if gate collision" in Eq. 4 with `r_T` penalty, but the detection mechanism is unspecified.
3. **"Gate collision" in the paper context means physically hitting the gate structure**, which is a sim-to-real concern, not a learning requirement.

Since the primary objective is learning to make progress along the track, gate collision detection is unnecessary and actively harmful.

**Fix:** Removed all gate collision logic:
- `quadrotor_env.cpp`: Removed `is_gate_collision_` flag, removed collision branch from gate-plane crossing (only gate pass detection remains), removed gate collision from `isTerminalState()`
- `quadrotor_env_test.cpp`: Replaced `CollisionOutsideOpening` with `OutsideOpeningNoEffect`, removed all `isGateCollision` assertions, renamed `DistantPlaneCrossingNoCollision` → `DistantPlaneCrossingNoEffect`
- Terminal conditions now: ground crash (z≤0.02), out of bounds (||p||>100), timeout (t≥max_t) — no gate collision

**Expected effect:** Episodes only terminate on physical crashes or timeouts. The agent is free to explore the track without spurious terminal events from gate plane crossings. Reward is purely progress-based.

**Outcome:** ✅ All 25 unit tests pass. _Training validation pending._

---

## 2026-05-01 — Bug H: Constant-offset deviation loophole in spline progress reward

**Hypothesis:** The spline progress score is `s(p) = arc_along_spline − 0.5 × lateral_deviation`. The per-step reward is `s(p_t) − s(p_{t-1})`. Because deviation is baked into the score function as a linear term, a drone flying **parallel** to the spline at a constant large offset (e.g. 12m) gets `Δdev ≈ 0` every step — the deviation terms cancel in the difference, and the reward equals pure forward arc-length progress. The 0.5 ratio only charges a one-time cost when the drone first drifts away, not for *sustaining* a large offset. This allows the agent to accumulate entirely positive reward while flying so far off-track that it misses gates entirely.

Confirmed visually in the reward debugger: dragging the drone in a path parallel to but ~12m away from the spline accumulated +42 cumulative reward with zero negative steps, despite being far past G1's opening.

**Fix (two parts):**

1. **Increased `spline_deviation_ratio` from 0.5 → 1.5:** Triples the one-time cost of drifting away from the spline (and the one-time gain of returning). This makes the initial drift more expensive but does NOT fix the constant-offset problem.

2. **Added per-step quadratic deviation penalty:** New reward component `r_dev = −coeff × max(0, dev − gate_width/2)²` fires **every step** the drone is beyond gate radius (gate_width/2 = 2.5m). Unlike the linear term inside `spline_progress()`, this cannot be cancelled by the difference — it's a standalone per-step cost. At 12m deviation (excess = 9.5m), penalty = `−0.01 × 9.5² ≈ −0.9` per step — easily overwhelming any forward progress gain.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp:158,164,174` — Added `deviation_penalty_coeff_` (default 0.01), `last_deviation_penalty_` tracking, `spline_nearest_dist()` helper
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:285,300-310,510,672-685,690` — Added per-step deviation penalty in `step()` after progress reward; added `spline_nearest_dist()` brute-force nearest-point distance; config loading for `deviation_penalty_coeff`; `r_deviation` in `extra_info`
- `flightlib/configs/quadrotor_env.yaml:136-137` — `spline_deviation_ratio: 0.5 → 1.5`, added `deviation_penalty_coeff: 0.01`
- `flightlib/configs/test_env.yaml:52` — added `deviation_penalty_coeff: 0.01`
- `flightrl/training/reward_math.js:7-8,128-140` — Mirrored C++ changes: `splineDeviationRatio: 1.5`, `deviationPenaltyCoeff: 0.01`, quadratic penalty in `computeRewardDelta()`
- `flightrl/training/reward_debugger.html:48,330-336` — Added deviation penalty display to HUD (turns red when beyond gate radius)

**Expected effect:** Drone can no longer accumulate positive reward while cruising far off-track. Staying within ~2.5m of the spline centerline incurs no penalty; beyond that, the quadratic cost grows rapidly. At 5m excess deviation the penalty is -0.25/step; at 10m excess it's -1.0/step. This forces the agent to stay near the racing line where it can actually navigate through gates.

**Outcome:** ✅ All 6 spline/reward tests pass. C++ builds clean. Python bindings rebuilt. _Training validation pending._

---

## 2026-05-01 — Bug I: Distributed init applied during eval/visualization, masking true policy quality

**Hypothesis:** Visualizations of the trained policy showed the drone apparently flying backwards away from G0 yet accumulating large positive cumulative reward (e.g. R=+17.5 over 94 steps before crashing). Tracing the same plotted path through the JS reward debugger yielded a strongly negative score, suggesting either a C++ reward bug or a visualization mismatch.

Investigation ruled out the reward math. The C++ telescoping update (`progress = s_curr - s_prev_; s_prev_ = s_curr`) and the gate-pass branch (`s_prev_ = spline_progress(pos, new_idx)`) are standard and correct. A diagnostic print of `info["gate_idx"]` on step 0 of every env revealed the actual cause:

- `QuadrotorEnv::reset(obs, random=true)` is the default ([quadrotor_env.hpp:94](flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp#L94)).
- `FlightmareSB3VecEnv.reset()` calls `wrapper.reset(obs)` with no args, so even eval used distributed init.
- Of 100 eval envs, **93 spawned mid-track with `current_gate_idx_ ∈ {1..10}`**; only 7 actually started at G0.
- `visualize_policy.py` reconstructs position by integrating velocity from a hardcoded `start_pos`, so the plotted path is anchored at the wrong origin whenever distributed init fires. The drone in the picture really *did* fly forward along its actual segment (e.g. segment 7); the renderer drew that velocity trace starting from `[17.4, -5, 6.31]`, making it look like the drone wandered backward through G0. The +17.5 reward was earned legitimately on the wrong segment.

The user wants eval/visualization to behave per the paper objective: spawn at `start_pos`, target G0, attempt the full track. Distributed init must remain on for *training* (it's one of the paper's three key ingredients, L247-261).

**Fix:** Added a runtime-toggleable `random_reset_` flag on `VecEnv` honored by both explicit reset and the in-step autoreset on done. Default stays `true` so training is unchanged; eval/visualization scripts flip it to `false` once before the rollout loop.

**Files changed:**
- `flightlib/include/flightlib/envs/vec_env.hpp:39-44,94-95` — Added `setRandomReset(bool)`/`getRandomReset()` and `random_reset_{true}` member
- `flightlib/src/envs/vec_env.cpp:74,160` — `envs_[i]->reset(obs.row(i), random_reset_)` in both explicit reset and `perAgentStep` autoreset
- `flightlib/src/wrapper/pybind_wrapper.cpp:27-28` — Bound `setRandomReset` and `getRandomReset`
- `flightrl/training/sb3_vec_env.py:88-94` — Added `set_random_reset(bool)` forwarder
- `flightrl/training/visualize_policy.py:185-188` — Calls `raw_env.set_random_reset(False)` before `env.reset()` in `record_episodes()`
- `flightrl/training/gate_racing_iros.py:255-259` — Calls `raw_env.set_random_reset(False)` in `evaluate()` before the rollout loop

**Outcome:** ✅ Confirmed. Post-fix diagnostic print shows all 100 envs report `gate_idx=0` on step 0. Honest eval of the current best model from `start_pos` reveals R≈-28 with only 1 gate passed per episode — substantially worse than distributed-init eval was suggesting, but a true measure of full-track policy quality.

**Note on stale binaries:** `flightrl/training/flightgym.cpython-38-aarch64-linux-gnu.so` is a copy that takes import precedence over the freshly built `flightlib/flightgym.cpython-38-aarch64-linux-gnu.so`. After any C++/pybind change, copy the rebuilt `.so` from `flightlib/` over the one in `flightrl/training/` — otherwise Python loads stale bindings and new methods raise `AttributeError`.

---

## 2026-05-01 — Bug J: Distributed-init audit (Phase 2 ω drop, naming, hardcoded floor)

**Hypothesis:** Audit of the two-phase distributed-init implementation against paper L247-261 plus a Python diagnostic ([flightrl/training/diag_distributed_init.py](flightrl/training/diag_distributed_init.py)) surfaced three small flaws:

1. **Phase-2 reset drops `snap.omega`.** The trajectory snapshot stores body angular velocity ([quadrotor_env.hpp:67](flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp#L67)) but the Phase-2 branch in `reset()` only assigned pos/vel/quat. Combined with `quad_state_.setZero()` at the top of `reset()`, this left ω=0 — partially recreating the hover-init pathology in high-speed regions that the paper warns against (L261).

2. **`pass_rate` is mis-named.** It's `total_gates_passed / total_episodes` = mean gates per episode, not a probability. With 11 gates the production threshold of `0.6` means "average <1 gate cleared per episode" — a weak interpretation of "reliably" (paper L259). Renaming clarifies the metric and prevents future tuning errors.

3. **`total_episodes_ > 200` was a magic number** in [quadrotor_env.cpp:55](flightlib/src/envs/quadrotor_env/quadrotor_env.cpp#L55), not exposed to YAML.

**Diagnostic results before fix:**
- Phase 1: ✅ all 11 segments uniformly hit (min/max ratio 0.87 over 3000 samples), spawn speeds hover-like (mean 0.42 m/s, max 0.76 m/s).
- Phase 2: ✅ engages once per-env episodes > 200, mean_gates_per_ep > threshold, and buffer > 100. Late-run spawn speeds bimodal (Q4: mean 3.16 m/s, max 11.64 m/s, 55% above 2 m/s — matching `traj_init_fraction=0.5`).

**Files changed:**
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:88-94` — added `quad_state_.w(0..2) = snap.omega(0..2) + small noise` to the Phase-2 reset block
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:55-67` — renamed local `pass_rate` → `mean_gates_per_ep`, gated transition on `traj_init_min_episodes_` member instead of `200`
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:511` — load `traj_init_min_episodes` from YAML
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp:143-144` — added `traj_init_min_episodes_{200}` member, updated comment on `traj_init_threshold_`
- `flightlib/configs/quadrotor_env.yaml:134-136` — corrected comment ("NOT a probability"), added `traj_init_min_episodes: 200`
- `flightlib/configs/test_env.yaml:51` — added `traj_init_min_episodes: 200`
- `flightrl/training/diag_distributed_init.py` — new diagnostic script that exercises both phases and reports verdicts

**Expected effect:** Phase-2 spawns now carry realistic angular velocity (no longer ω=0 immediately after a flying-state replay). Code is self-documenting on the metric semantics. Operators can tune the Phase-2 transition without recompiling.

**Outcome:** ✅ Confirmed. C++ builds clean. Re-run of diagnostic post-fix: Phase 1 still passes; Phase 2 still engages (log message now reads `mean_gates_per_ep=...`); late-run spawn speeds remain bimodal (Q4: mean 3.18 m/s, max 18.06 m/s, 42% above 2 m/s). No regression. Open follow-up: the production `traj_init_threshold=0.6` (avg gates/ep) is likely too high for the current saved policy (~0.18 avg), meaning Phase 2 may never engage in real training runs — verify with TensorBoard's `gates_passed` metric and consider lowering once we have post-25M-step data.

---

## 2026-05-01 — Bug K: G1 perpendicular to approach + sharp turn into G2

**Hypothesis:** Visual inspection of the trajectory plot showed the drone struggling at G1. Decoding the gate quaternions confirmed the issue:
- G0[17.25, 3.63, 6.31] yaw ≈ 84° → normal points roughly +y (drone exits going north).
- G1[16.75, 39.57, 6.76] yaw ≈ 177° → normal points -x. The G0→G1 segment is a 36 m straight line in +y, so the drone arrives at G1 with its velocity nearly *perpendicular* to G1's opening. Passing requires a 90°+ in-place yaw, which the agile-but-light drone cannot execute mid-flight.
- After G1, the path immediately turns ~70° back to G2[2.33, 27.87, 2.55] (SW), making the G0→G1→G2 transition a hairpin rather than a curve.

**Fix (track design, no code change):**

1. **Re-orient G1** to yaw ≈ 100° — normal mostly +y with a slight -x pre-rotation toward G2. Drone now enters G1 along its approach direction with only ~20° of misalignment instead of 90°+.
   - New quat: `[0.6428, 0.0, 0.0, 0.7660]` (pure yaw rotation).
2. **Insert a new gate** at `[10.0, 33.0, 4.5]` between G1 and G2, oriented along the curve tangent (yaw ≈ -141°, pitch 12.8°).
   - Quat: `[0.3318, 0.1043, 0.0369, -0.9367]`.
   - Splits the ~70° G1→G2 turn into two ~35° turns, giving the cubic Hermite spline a smoother arc to interpolate.

Track now has 12 gates (was 11). `num_gates_ = gates_.size()` is dynamic — no C++ change needed; the spline builder, gate-pass detector, and obs construction all handle the new gate count transparently.

**Files changed:**
- `flightlib/configs/quadrotor_env.yaml:64-75` — replaced G1's quaternion; inserted new gate between G1 and G2.
- `flightrl/training/reward_math.js:14-15` — synced JS reward debugger to match the new track.

**Expected effect:** First-segment success rate should improve since G1 no longer demands a 90° yaw mid-flight. The G1→G2 transition becomes a gentle curve rather than a hairpin, reducing the lateral deviation penalty that was likely dominating reward in that region. Note: existing trained policies will need to be re-trained — the 12-gate spline is a different track, and the 2-future-gate observation now references different downstream gates.

**Outcome:** ✅ Build + integration validated. C++ rebuilds clean and the `.so` was copied to `flightrl/training/`. Env logs confirm `Loaded 12 gates with normals` and `Built spline progress tracker: 12 segments, total arc length = 209.901718 m` (down from 213.0 m on the 11-gate version — the inserted gate cuts the G1→G2 corner, which is the intended geometric effect). All 5 `SplineProgress.*` unit tests pass; the only failing tests (`EigenChecks.EigenQuaternionSequence`, `EigenChecks.QuaternionCrossMatrix`, `QuadrotorEnv.Constructor` with stale `OBS_DIM=12`) were verified to fail identically on the prior commit, so they are pre-existing and unrelated. `diag_track.py` confirms `random_reset=False` spawns all 100 envs at `[17.4, -5.0, 6.31]` with `gate_idx=0`, and a 200-step random-action rollout produces realistic per-step rewards (range −0.65 to +0.01) and velocities (mean 7 m/s, max 12.8 m/s). SB3 wrapper smoke test passes — env+wrapper return correct shapes and infos. _Training validation pending a fresh PPO run._

---

## 2026-05-01 — Bug L: Redundant brute-force nearest-point search halving training FPS

**Hypothesis:** Training FPS dropped from ~28k to ~13k after the spline reward landed (Bug F + Bug H). The cause is in `step()`: every step, every env now runs **two** brute-force nearest-sample scans over the 200-sample-per-segment spline. `spline_progress()` does a 200-iter loop to compute `best_dist_sq`, then `spline_nearest_dist()` (called immediately after for the per-step deviation penalty) runs the *same* loop again over the same samples to recover the same distance. That's ~2× the per-step C++ work compared to the pre-spline linear `segment_progress()`, and it doubles the most expensive part of the post-spline path.

**Fix:** Add an out-param `Scalar* out_raw_dist = nullptr` to `spline_progress()` that captures `sqrt(best_dist_sq)` *before* the neighbor-projection refinement. Bit-identical to what `spline_nearest_dist()` returns (which itself does no refinement) — no functional reward change. `step()` reads the out value instead of calling `spline_nearest_dist()`. The early-return paths (invalid gate idx, empty samples) write 0 to the out-param at function entry, matching the standalone function's `return Scalar(0)`.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp:176-177` — added optional `Scalar* out_raw_dist = nullptr` parameter to `spline_progress()`; left `spline_nearest_dist()` declaration in place (still public, unused after this change but kept for API stability)
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:628-660` — write `sqrt(best_dist_sq)` to `*out_raw_dist` after the brute-force loop and before the refinement loop; initialize `*out_raw_dist = 0` at function entry to preserve the early-return semantics
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:300-310` — `step()` now declares `Scalar dev_raw = 0`, passes `&dev_raw` to `spline_progress()`, and uses `dev_raw` directly in the deviation-penalty branch instead of calling `spline_nearest_dist()`

**Expected effect:** ~halves the per-step nearest-point work in the C++ env. Should recover most of the 28k → 13k FPS regression without any change to the reward signal. Reward gradient, optimal policy, learning curve all unchanged — only wall-clock speed changes.

**Outcome:** ✅ Build clean, `.so` copied to `flightrl/training/`. All 5 `SplineProgress.*` unit tests pass. Bit-identical reward verified by code inspection: the value written to `*out_raw_dist` is `std::sqrt(best_dist_sq)` computed from the same brute-force scan as `spline_nearest_dist()`, before any refinement. _Training FPS validation pending a fresh PPO run._

---

## 2026-05-01 — Experiment M: Disable distributed initialization (test always-start-from-start)

**Hypothesis:** The IROS 2021 paper says distributed init is one of three "key ingredients", but with the spline-based progress reward + per-step deviation penalty (Bug 9), the policy may actually learn faster from a fixed start: the early gates get many more attempts per unit wall-clock, the obs distribution is narrower (better for VecNormalize), and the policy avoids spending the first ~5M steps learning to hover-recover from random orientations near far-track gates. User wants to A/B this against the current distributed-init run.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp` — added `bool distributed_init_{true};` member
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` — `reset()` now gates the Phase 1/Phase 2 init block on `distributed_init_`; the `else if (random)` no-gates branch tightened to `else if (random && num_gates_ == 0)` so that `random && num_gates_>0 && !distributed_init_` falls through to the deterministic `start_pos_` reset. `loadParam()` reads `cfg["rl"]["distributed_init"]`.
- `flightlib/configs/quadrotor_env.yaml` — added `distributed_init: false` (default flipped to off for this experiment; set back to `true` to restore paper behavior)

**Expected effect:** Every env now resets to `start_pos_ = [17.4, -5.0, 6.31]` upright at hover, current_gate_idx=0. Gate 0 should be cleared earlier in wall-clock terms; later gates may take longer to learn (no curriculum from far-segment hover). If this run reaches gate ≥3 faster than the distributed-init baseline, the paper's key-ingredient claim doesn't apply with our reward shaping.

**Outcome:** _pending — flag wired, rebuild + training run TBD_

## 2026-05-01 — Bug N: Curriculum rung 0 — drone dives into ground after partial forward progress

**Hypothesis:** Trajectory plot from rung 0 best_model (single_1, start z=5, gate z=5) shows the drone reaching ~half the segment then crashing at z=0 with chaotic bang-bang motor commands across all 4 thrusts. Root cause: the reward is misshaped for stability —
1. Progress reward is unbounded along the spline → policy locally maximizes by lurching forward.
2. Per-step `r_deviation` has a 2.5 m deadzone — drone can fall ±2.5 m below spline z=5 at zero cost; by the time the penalty bites, it has too much downward velocity to recover.
3. `body_rate_coeff` was 0.0005 with anneal-to-0 over 5 M steps → at 5 M training steps the rotational-stability shaping is fully gone, so the policy outputs bang-bang motor commands without penalty.
4. No action-smoothness term anywhere.

The drone has learned a degenerate strategy: tip forward, pile on thrust, accept the crash. RL working as designed — the reward is wrong, not the optimizer.

**Fix (YAML-only, no recompile, this commit):**
- `flightlib/configs/quadrotor_env.yaml:65` — `body_rate_coeff: 0.0005 → 0.005` (10×)
- `flightlib/configs/quadrotor_env.yaml:66` — `body_rate_anneal_steps: 5_000_000 → 100_000_000` (effectively never anneal)
- `flightlib/configs/quadrotor_env.yaml:75` — `deviation_penalty_coeff: 0.2 → 2.0` (10×)

**Expected effect:** With 10× stronger off-spline penalty plus a 10× harder rotational-chaos cost that no longer anneals away, the drone should learn smooth coordinated thrust to stay near the spline z=5 line. Transfer-learning from current `best_model.zip` (which already knows "fly forward") + new shaping should converge in 1.5–2 M extra steps; if not, escalate to C++ changes (remove deadzone in dev penalty, add z-specific term, action-smoothness `‖a_t − a_{t−1}‖²`).

**Outcome:** _pending — fine-tune in progress_

## 2026-05-01 — Bug J: Best-model save overwritten by short follow-up runs

**Hypothesis:** `TrainingRewardCallback.__init__` resets `best_mean_reward = -np.inf` every run. The very first PPO iteration's `mean_reward` (computed over 100 episodes of a barely-warm policy) is therefore always strictly greater than `-inf`, and gets saved as the new "best" — overwriting any genuinely-trained model that lived at the same `--save-path` from a prior run. We discovered this when `saved/curriculum/03_bent_3_15/best_model.zip` evaluated at 0/20 gates (mean reward `-2.91`) despite the user reporting that bent_3_15 had successfully learned in an earlier 3M-step run. The killer was `rung3_v6.log` — killed at 150K steps with `★ New best reward: -92.92` written, clobbering the v2/v3/v5 trained policy.

**Files changed:**
- `flightrl/training/gate_racing_iros.py:60-65` — added `min_steps_before_save=300_000` parameter to `TrainingRewardCallback.__init__`
- `flightrl/training/gate_racing_iros.py:114-122` — gate the best-model save on `self.num_timesteps >= self.min_steps_before_save`

**Expected effect:** A run killed in its first ~2 PPO updates can no longer overwrite a prior good model. Save threshold of 300K is well below the natural ramp-up of any meaningful learning, so it doesn't delay legitimate improvements. This is a defensive fix — the user should still pass per-rung `--save-path` to keep curriculum saves isolated.

**Outcome:** _pending — first real test is the bent_3_45 retraining about to start_


## 2026-05-01 — Bug K: Survival-bonus magnitude > spline progress on hard tracks

**Hypothesis:** With `survival_bonus: 0.05` per step at 50 Hz, surviving the full 30s episode yields +75 reward — much larger than the ~24 reward available from completing the 24 m bent_3_45 spline. So once the drone fails to make the 45° turn cleanly, the optimal policy is to fly straight up to a safe altitude and accumulate survival reward instead of attempting the gates. Confirmed on bent_3_45 best_model: deterministic eval gates_passed=1/3 in 20/20 episodes, trajectory clears G0 then climbs to z≈50 m and out to x≈45 m. Same pattern previously seen on circle_4. Bent_3_15 happened to learn through this because the 15° turn was easier than the climb-out hack — but the reward shape is structurally wrong for any track where the turn is non-trivial.

**Files changed:**
- `flightlib/configs/quadrotor_env.yaml:69` — `survival_bonus: 0.05 → 0.005` (10× reduction). 30s of survival now yields +7.5 reward, smaller than the ~24 reward of completing the spline, so progress should dominate.

**Expected effect:** Drone still has a small alive-floor (so it doesn't immediately give up and crash like the no-survival circle_4 attempt did), but the floor is small enough that gate-following becomes the dominant reward source. If this still produces falling, escalate to option 2 (z-cap penalty) or option 3 (alive-only-near-spline).

**Outcome:** _pending — retraining bent_3_45 from scratch with this setting_



## 2026-05-01 — Bug L: Re-add crash_penalty for circle_8 (drone not learning altitude)

**Hypothesis:** On circle_8 the drone is making good lateral progress around the track but keeps crashing into the ground — i.e. it has learned the gate-following geometry but has no signal pushing it to maintain altitude. `crash_penalty` was stripped in Bug B (2026-04-30) for paper-faithfulness, leaving terminal reward = 0 on ground crash. With pure progress reward and no terminal cost, an early ground impact and a successful gate pass differ by only the few steps of progress lost — too weak a gradient for "stay airborne" to dominate. A discrete `-5` terminal penalty (2× a typical gate's progress reward) gives the policy a clear cost for crashing without overwhelming the per-step shaping.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp` — added `crash_penalty_` member (default 0.0) and `last_crash_penalty_` for diagnostics
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `loadParam()` — read `rl.crash_penalty` from YAML
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `isTerminalState()` — apply `crash_penalty_` on ground crash (`z ≤ 0.02`) and OOB (`||p|| > 100`); timeout still returns 0
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `updateExtraInfo()` — expose `r_crash_penalty` for diagnostics
- `flightlib/configs/quadrotor_env.yaml:99` — `crash_penalty: -5.0` (now actually wired through)

**Expected effect:** Resuming from the 8M-step circle_8 checkpoint, episode-length distribution should shift upward (fewer short ground-impact episodes). `r_crash_penalty` in extra_info should be ≈0 for non-terminal steps and -5 for ground/OOB terminations. If the policy was reward-hacking by skimming low, it should learn to climb. Watch for the opposite failure mode (climbing to ceiling to "play it safe") — if it appears, the penalty is too strong relative to progress on circle_8 (~24 reward).

**Outcome:** _pending — resume training from 8M-step checkpoint_


## 2026-05-01 — Bug M: Add per-step floor penalty + raise track to z=6m (curriculum aid)

**Hypothesis:** After Bug L's `crash_penalty=-5` was added, circle_8 training showed the policy clearing 3 gates around the loop but `ep_rew_mean` started declining — terminal -5 at z=0 is too sparse a signal for "stay airborne". A drone skimming low only sees the cost on the rare step it actually impacts the ground; in between, no gradient pushes it up. The fix is a per-step linear floor penalty `r_floor = -coeff × max(0, threshold − z)` that fires every step the drone is below `threshold`, making low-altitude flight continuously costly. Pairing this with a higher track (z=6m gates instead of z=3m) gives the drone 4m of headroom above the 2m floor instead of 1m, so a brief altitude dip doesn't immediately enter the penalty zone. Both shaping terms are designed to be removed once the policy reliably stays above 2m.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp` — added `floor_penalty_coeff_`, `floor_threshold_`, `last_floor_penalty_`
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `loadParam()` — read `rl.floor_penalty_coeff` and `rl.floor_threshold` from YAML
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `step()` — apply `−coeff × (threshold − z)` when `z < threshold`
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` `updateExtraInfo()` — expose `r_floor_penalty`
- `flightlib/configs/quadrotor_env.yaml` — `start_pos` z=3→6, all 8 circle_8 gates z=3→6, added `floor_penalty_coeff: 2.5` and `floor_threshold: 2.0` (peaks at -5/step at z=0, zero at z≥2)

**Expected effect:** Resuming from the post-Bug-L checkpoint, drone should learn within ~1-2M steps to keep z ≥ 2m. Watch `r_floor_penalty` in extra_info: should start large-negative on episodes where the drone dips low, then decay toward 0 as the policy adapts. Risk: at -125 reward/sec at z=1m (vs ~24 reward for full track), the floor cost may dominate progress and cause the policy to climb to the world ceiling instead of racing — if that happens, drop coeff from 2.5 → 1.0 (peak -2 at z=0). Note the YAML obs distribution shifts when gates move from z=3 to z=6: VecNormalize running stats will adapt over ~500K-1M steps.

**Outcome:** _pending — resume training from current best_iros_gate_racing checkpoint_

---

## 2026-05-01 — IL Warm-Start: BC pretraining of policy + value head

**Hypothesis:** RL from scratch fails on this setup because random per-motor actions only produce crashes; the agent never discovers "fly toward gate" behavior. Pre-training the same MLP via behavioral cloning on a classical trajectory-tracker's demonstrations gives PPO a prior that already flies — fine-tuning then refines, not bootstraps. Followed `IL_WARMSTART_PLAN.md` but with three additions found necessary while debugging:

1. **Pre-train the value network too** (not just the policy). PPO's first update uses advantages from `V(s)` — if the value head is randomly initialized, the advantage signal is garbage and the LR=3e-5 PPO update wrecks the BC policy on the very first rollout. Fix: jointly minimize `MSE(action, expert_action) + 0.5 * MSE(value, MC_return)` using discounted returns computed per demo file.
2. **Shrink `policy.log_std`** before saving. SB3 default is `log(1)=0` → stochastic action stddev = 1.0 in `[-1,1]` space. PPO's training rollouts are stochastic, and with stddev=1 the actions are pure noise; the policy collapses immediately. Fix: set `log_std = log(0.1)` so early stochastic samples stay close to the BC mean.
3. **Disable `distributed_init`** for BC warm-start. Demos come from `start_pos`; if training spawns drones at random track segments, the BC policy is OOD there and PPO immediately destroys it. Fix: `distributed_init: false` until end-to-end policy is solid, then optionally flip back on.

**Files changed:**
- `flightrl/training/trajectory_controller.py` — read `start_pos`, gate positions, mass, inertia from `quadrotor_env.yaml` instead of hardcoding (z=3 vs env's z=6 mismatch made existing demos miss every gate)
- `flightrl/training/train_bc.py` — new BC script: joint policy+value training, MC-return targets, best-val early stopping, input noise (DART-style), log_std shrink, `--verify` mode that runs the trained policy in env
- `flightrl/training/gate_racing_iros.py:447-451` — added `--learning-rate` flag to override PPO LR after `--load-model` (recommended 3e-5 to preserve BC prior)
- `flightlib/configs/quadrotor_env.yaml:105` — `distributed_init: true → false`
- `flightrl/training/demos/circle_8_slow.npz`, `circle_8_med.npz` — fresh demos at 3.5 / 4.5 m/s (8/8 gates each, 3102 transitions total)

**Expected effect:** BC alone passes 2 gates deterministically (validated: 10/10 episodes, R=25.3, ep_len=468). PPO fine-tune at LR=3e-5 shows monotonic reward growth instead of collapse (validated on 1.5M-step smoke test: -6.2 → +4.78 across 10 PPO updates). Full 25M-step run should clear the circuit.

**Workflow:**
```bash
cd ~/flightmare/flightrl/training
# 1. Collect demos
python trajectory_controller.py --speed 3.5 --save-path demos/circle_8_slow.npz
python trajectory_controller.py --speed 4.5 --save-path demos/circle_8_med.npz
# 2. BC pretrain (joint policy+value, ~30s on CPU)
python train_bc.py --demos demos/circle_8_slow.npz demos/circle_8_med.npz \
  --save-dir saved/bc_pretrained --epochs 200 --verify
# 3. PPO fine-tune (low LR preserves BC prior)
python gate_racing_iros.py --train 1 \
  --load-model saved/bc_pretrained/bc_model.zip \
  --load-vecnorm saved/bc_pretrained/vec_normalize.pkl \
  --save-path saved/bc_then_ppo --tb-log saved/tb_logs/bc_then_ppo \
  --learning-rate 3e-5 --total-timesteps 25000000
```

**Outcome:** _pending full 25M-step fine-tune — IL chain validated end-to-end on 1.5M smoke test_

---

## 2026-05-01 — Bug O: Remove floor penalty (Bug M curriculum aid)

**Hypothesis:** The per-step floor penalty added in Bug M was always flagged as a curriculum aid to be removed once the policy reliably stays above 2 m. With the YAML coefficient already at 0, the gated branch in `step()` is dead code that costs a per-step compare and clutters the reward bookkeeping. With the IL warm-start (`circle_8_slow.npz` / `circle_8_med.npz` demos at z≈6 m) plus `crash_penalty=-5`, altitude maintenance is no longer the load-bearing signal that needed shaping. Removing the entire mechanism — member fields, step branch, YAML keys, `extra_info` entry — restores paper-faithful reward shape and avoids the foot-gun of a future re-enable that nobody remembers wiring through.

**Files changed:**
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp` — removed `floor_penalty_coeff_`, `floor_threshold_`, `last_floor_penalty_` members
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` — removed `last_floor_penalty_` reset in `step()`, removed the floor-penalty branch (formerly section 5; renumbered survival bonus 5→5), removed `floor_penalty_coeff` / `floor_threshold` YAML loading in `loadParam()`, removed `extra_info_["r_floor_penalty"]`
- `flightlib/configs/quadrotor_env.yaml` — removed `floor_penalty_coeff` and `floor_threshold` keys

**Expected effect:** Reward identical to before (coeff was already 0 ⇒ branch never fired). Slight FPS gain from one fewer per-step compare. Logs no longer carry the dead `r_floor_penalty=0.0` field. Any downstream analysis script that read `info["r_floor_penalty"]` will KeyError — none found in repo at time of removal (`grep -r "r_floor_penalty"` returns 0 hits).

**Outcome:** _pending — needs C++ rebuild + `.so` copy to `flightrl/training/` before next training run_

---

## 2026-05-01 — Throughput tuning: 100 envs × 1500 steps → 200 envs × 750 steps

**Hypothesis:** More parallel envs increases init-state diversity per PPO update (each env starts from a distinct distributed-init point), which should improve coverage of the spline rather than "explore harder" in the action-noise sense. Halving `n_steps` keeps the rollout batch at 150K so PPO hyperparameters (`batch_size`, `n_epochs`, `learning_rate`) don't need retuning. The 8-core ARM VM caps how much parallelism actually helps — measured before committing.

**FPS scan with random-action benchmark** (`/tmp/bench_fps.py`, 500 steps × N envs through the SB3 wrapper):

| envs | FPS    | per-step | speedup vs 100 | parallel efficiency |
|------|--------|----------|----------------|---------------------|
| 100  | 161K   | 0.6 ms   | 1.00×          | 100%                |
| 200  | 238K   | 0.8 ms   | 1.48×          | 74%                 |
| 300  | 277K   | 1.1 ms   | 1.72×          | 57%                 |
| 400  | 318K   | 1.3 ms   | 1.97×          | 49%                 |

**Files changed:**
- `flightlib/configs/vec_env.yaml` — `num_envs: 100 → 200` (no rebuild needed; loaded at runtime)
- `flightrl/training/gate_racing_iros.py:191` — `n_steps=1500 → 750` (200 × 750 = 150K, unchanged batch size)

**Why 200, not 300/400:** 200 is the last point on the curve where the marginal env still buys ≥0.7 of an env's worth of throughput. Past 300 we're paying compute for shrinking returns; the diversity gain is real but throughput saturates. Also 750 steps × 0.02s = 15s of sim time per rollout, which still fits 1-3 mid-training episodes per env without excessive bootstrap reliance (dense per-step reward from spline progress means truncation cost is small).

**Expected effect:** ~1.5× wall-clock speedup at unchanged update size. Slightly higher GAE bootstrap reliance at rollout boundaries (acceptable with dense reward). Same PPO learning dynamics — `ep_rew_mean` curve should look similar shape, just compressed in wall time. If it doesn't, the suspect is reduced rollout horizon causing value-function bias, not the env count.

**Outcome:** _pending — to be measured by comparing `time/fps` and `ep_rew_mean` slope vs prior 100-env runs_


---

## 2026-05-02 — Curriculum sweep (paper-strict reward, ~42M total steps)

**Hypothesis:** Train each track in [TRACK_CURRICULUM_PLAN.md](TRACK_CURRICULUM_PLAN.md) sequentially with paper-faithful reward (no crash_penalty, no survival_bonus, no gate_bonus). Fine-tune from prior rung's best_model. Investigate ent_coef as a knob for breaking face-plant local optima per user instruction. Goal: pass every track's success criterion.

**Files changed:**
- `flightlib/configs/quadrotor_env.yaml` — set `gate_bonus: 5.0 → 0.0`, normalized comments to indicate paper faithfulness on `act_coeff`, `survival_bonus`, `crash_penalty`. Per-rung `start_pos` and `gates` cycled by `track_loader.py apply <name>`.
- `flightrl/training/gate_racing_iros.py:454-459` — added `--ent-coef` CLI flag (default 0.01); wired through both fresh-train and `--load-model` paths so the entropy coefficient can be overridden per rung without editing the script.
- `flightrl/training/curriculum_log.md` — new file, full per-rung writeup.
- `flightrl/training/curriculum_eval.sh` — new helper that runs deterministic 20-episode eval and grep-friendly summary.
- `flightrl/training/saved/curriculum/{rung00_single_1,rung01_straight_2,rung02_straight_3,rung03_bent_3_15,rung04_bent_3_30,rung06_circle_8,rung10_vertical_3}/` — best_model.zip + vec_normalize.pkl + checkpoints + eval.log per rung.

**Outcome:** Mixed — a clear ceiling pattern, plus one false lead.

- ✅ **Passed (5):** rung 0 (`single_1`), rung 1 (`straight_2`), rung 2 (`straight_3`), rung 3 (`bent_3_15`), rung 10 (`vertical_3`). All 20/20 deterministic, full lap.
- ⚠️ **Failure rung (per plan §3): rung 4 (`bent_3_30`).** 8M steps (2× budget) failed to break the 2/3 gates plateau. The drone hits gate 1 (heading 30° from start, no problem) but then can't bank hard enough to redirect to gate 2 at heading 60° — overshoots laterally and crashes around step 125 (2.5s). Same plateau across `ent_coef ∈ {0.01, 0.015, 0.02, 0.03}`.
- ❌ **Sustained-turning fail: rung 6 (`circle_8`).** 8M steps; drone passes G0 of the circle then ground crashes. Confirms the rung-4 ceiling is not specific to that track — it's the policy's inability to commit to ≥30° per-gate turning under paper-strict reward + 0.73 kg / paper-inertia drone with 5m gate.
- ✅ **Vertical motion is fine: rung 10 (`vertical_3`)** passed in 2M steps. Confirms the failure mode is specifically about *yaw*, not 3D control in general — the drone can climb at 7° pitch through 3 gates without trouble.

**Key empirical finding (deviates from user's prescription):** `ent_coef` bumps helped on the straight-3 plateau (rung 2) but *hurt* on bent_3_15 (rung 3). On bent tracks the policy needs to commit to a single bent trajectory; high entropy keeps it oscillating and unable to settle. Default `ent_coef=0.01` plus more iterations (12M total on rung 3) was the actual unlock. User's "increase entropy if face-planting" hypothesis is correct for *exploration*-bottlenecked rungs but the wrong tool for rungs where the policy needs to converge on a non-straight trajectory.

**Why the failure rung exists:** three plausible contributing factors documented in `curriculum_log.md` § "Why training stalled at the bend ceiling":
1. The 4M-step per-rung budget is below paper's 25M; PPO from-scratch on a 30°-bend track may need much more.
2. Wide gates (5m) make "skim past at the wrong angle" a dominant strategy that doesn't transfer through curriculum.
3. Paper-strict reward (no gate_bonus, no crash_penalty) creates only progress-derivative gradient — no step-function signal at gate boundaries.

**Recommended follow-up experiments** (in `curriculum_log.md`): higher-resolution rung-4 characterization, smaller gate_width, small gate_bonus reintroduction, IL warm-start from trajectory_controller demos on bent tracks.

---

## 2026-05-02 — Curriculum extended-budget retry (rung 4 onward, ≥20M/rung)

**Hypothesis:** The "failure rung" verdict on bent_3_30 (rung 4) and circle_8 (rung 6) was premature. Both runs had `ep_rew_mean` still climbing when training was cut at 8M steps (2× the 4M plan budget). The plan §3 plateau check is "soft signal stays flat for the full extended budget" — neither run actually plateaued by that test; logs show ep_rew_mean still rising (rung 4 v2 hit ep_rew_mean=15.8 at cutoff; rung 6 v2 hit ep_rew_mean=16.1 at cutoff). User direction: train each post-rung-3 track with at least 20M steps (more for harder rungs like circle_8), and don't stop while reward is still climbing. Continue the curriculum sequentially.

**Plan:**
1. Resume rung 4 (bent_3_30) from `saved/curriculum/rung04_bent_3_30/best_model.zip` for 20M additional steps. Save to `rung04_bent_3_30_v3` to preserve the prior model. Don't cut early if ep_rew_mean trend is positive.
2. If rung 4 clears, advance to rung 5 (bent_3_45) with 20M+ from rung 4.
3. Then rung 6 (circle_8) with 25M+ from rung 5 (sustained turning is the hardest known case — give it the most budget).
4. Continue through rungs 7 (circle_6), 8 (circle_4), 9 (oval_8) at 20M+ each.
5. Append per-rung outcomes to `flightrl/training/curriculum_log.md`.

**Files changed (so far):**
- `flightlib/configs/quadrotor_env.yaml` — `track_loader.py apply bent_3_30` cycled the track to rung 4 geometry. Reward / dynamics block unchanged from the prior sweep (paper-strict).

**Expected effect:** Hitting the success criteria for at least bent_3_30 (full lap in ≥16/20) and ideally pushing through to circle_8. The training reward trajectory at the prior cutoff (still rising at iteration 27) suggests there's still optimization headroom. If reward genuinely plateaus mid-run, that becomes the documented true failure rung — but only after a real budget.

**Outcome:** _pending — first extension run launched._
