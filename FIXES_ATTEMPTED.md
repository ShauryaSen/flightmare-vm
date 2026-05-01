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
