# Experiment Plan — Bootstrapping RL for Drone Racing: A Three-Way Comparison

**Status:** Draft v3 · 2026-05-07
**Project:** Robot Learning class project / paper
**Scope:** ~9–13 hr of GCP compute (~$5–15 of $50 class credit) plus local dev/smoke time

> **Changes from v2 (2026-05-07):**
> - **Compute moves to GCP `c3-standard-8` on-demand** (8 dedicated vCPU x86, 32 GB RAM). Local UTM ARM64 VM relegated to dev/smoke only. Full §13 added with provisioning, build, snapshot, run, and cost-discipline workflow. Projected total spend ~$5–15 of the $50 class credit.
> - Throughput projection updated: **40K FPS conservative** (vs 10K on local), so the equal-budget 100M × 9 runs collapse to ~6.3 hr of training instead of ~25 hr. Whole campaign now fits in a single working day.
> - Implementation checklist (§8) restructured into local-prep → cloud-migration → run → wrap-up phases so the cloud handoff has explicit gates.
>
> **Changes from v1 (2026-05-07):**
> - Locked hyperparameters now reflect the **Optuna sweep winner** (Stage 2: 7/8 det at 50M from D_bent_3_30). High entropy (`ent_coef=0.05`) and small-rollout/aggressive-clip are surprises and are non-negotiable knobs the sweep selected.
> - Hand-designed curriculum (Condition B) is now a **concrete 6-rung ladder** explicitly chosen so each rung is empirically reachable from the previous one with the locked HPs, with circle_8 as the target endpoint. Figure_8 is moved to an **optional stretch rung** because the prior overnight chain hit a structural 6/8 wall that survived HP/ent/distributed-init/mini-track interventions.
> - Compute budget rebalanced: each condition gets an **equal 100M-step budget** rather than 25M (25M is insufficient for the curriculum to traverse 6 rungs, and the no-curriculum baseline deserves a fair shake at the same total). 100M was chosen to give the circle_8 rung enough room to fully solve (8/8) with margin, so the comparison isn't bottlenecked by the target.
> - Held-out track set updated to ones not in the new ladder.

---

## 1. Research Question

> **Is some form of curriculum necessary to train a PPO drone-racing policy, and when curriculum is used, does the choice between hand-designed and automated curriculum represent a tradeoff between specialization and generalization?**

## 2. Hypotheses

- **H1 (curriculum is necessary).** PPO trained from scratch on the target track within a fixed compute budget (75M env steps) fails to reach competent performance (≤ 1 gate cleared deterministically on circle_8).
- **H2 (specialization tradeoff).** A hand-designed structural curriculum is *more* sample-efficient on the target track topology — it injects prior knowledge about which difficulty axes (turning, altitude, speed) matter — but the resulting policy *overfits* the training distribution.
- **H3 (generalization tradeoff).** An automated random-track curriculum (IROS21-style) is *less* sample-efficient on the in-distribution target but produces policies that *generalize* better to held-out tracks because the agent sees a wider distribution of geometric parameters during training.

A clean falsification of any of H1–H3 is still a publishable class-project result.

---

## 3. Three Conditions

Each condition trains to a **fixed 75M-env-step budget** (see §6) and is evaluated under §7.

### Condition A — No Curriculum (baseline)
PPO from scratch directly on the target track (`circle_8`), 100M env steps. Same locked hyperparameters (§4) as B and C. Distributed initialization enabled (paper-faithful). Expected to fail; serves as the necessity test for H1.

### Condition B — Hand-Designed Structural Curriculum (6 rungs to circle_8)

Sequential fine-tuning across rungs, **carrying forward both policy weights and `VecNormalize` running stats**. Per-rung step budget held constant within the rung type (no per-rung HP tuning, no exploration kicks, no entropy schedules). Locked hyperparameters from §4 across every rung.

This ladder was chosen because each transition is **empirically demonstrated to work** in the prior `curriculum_v2/` runs:

| # | Rung | Track | Steps | Why this rung |
|---|---|---|---|---|
| 1 | A | `straight_2` | 10M | Warm start: 2 gates in a line. Establishes basic flight. (≥10M floor per prior overnight memory.) |
| 2 | B | `straight_3` | 10M | 3 gates straight. Forces longer-horizon planning. |
| 3 | C | `bent_3_15` | 12M | Introduces turning at gentle 15°. |
| 4 | D | `bent_3_30` | 13M | Steeper 30° turn. The hardest rung in the prior chain. |
| 5 | E | `bent_3_45` | 15M | 45° turn — bridges to closed-loop curvature of circle_8. New rung relative to v2 chain (smooths the bent_3_30→circle_8 jump that needed 50M last time). |
| 6 | F | `circle_8` | 40M | **Target rung.** 8 gates in a closed circle, full lap. The prior single-rung jump from D_bent_3_30 → circle_8 with locked sweep HPs reached 7/8 in 50M; with the extra E_bent_3_45 bridge plus 40M, expect ≥7/8 reliably across all 3 seeds with shot at 8/8. |
| **Total** | | | **100M** | |

**Optional stretch rung G — `figure_8` (not in budget):** The prior chain hit a hard 6/8 plateau on figure_8 that survived entropy bumps, HP sweep, distributed init, mini-tracks, and gate_bonus shaping. Adding it inside the 100M budget steals from circle_8 and risks all three seeds failing the *primary* result. If circle_8 finishes ≤ rung F's budget with margin (e.g., reaches 8/8 by 30M), run figure_8 as a stretch with the leftover steps and report whatever it reaches. Do not let figure_8 outcomes gate the paper.

**Critical change from prior practice:** use **identical hyperparameters across all rungs** (no per-rung `ent_coef` adjustments, no exploration kicks, no LR schedules). The current `saved/curriculum_v2/` results were produced with iterative per-rung tuning — that data is *not* usable for the paper (see §10).

### Condition C — Automated Random-Track Curriculum (IROS21)

Implements Song et al. IROS21 §IV "Random Track Curriculum": same locked hyperparameters; track topology family parameterized by per-gate position offsets `δ ∈ N(0, σ²·I)` and per-gate yaw offsets, sampled afresh every N episodes.

**Concrete sampler spec:**
- Base topology: `circle_8` (8 gates, 18 m radius, z=15) — matches target.
- Per-episode resample: at every env reset, sample a fresh track from the current σ. Each of the 100 parallel envs gets an independent draw, so the policy sees ~100 distinct tracks per rollout.
- Per-gate perturbation: `pos_perturbation = N(0, σ²·I)` applied to each gate's `(x, y, z)`. Gate yaw perturbed by `N(0, (10°·σ/σ_max)²)`. Gate sequence ordering and topology (closed loop) preserved.
- Spline rebuilt per draw using existing `buildTrackSpline()` (start_pos held fixed; gate normals derived from perturbed yaw).

**Adaptive σ controller:** start at σ=0.2 m. Maintain a rolling window of the last 50 episode-final `gate_idx` values across all envs. Compute pass-rate as `mean(gate_idx) / num_gates`. Every 100 episodes:
- If pass-rate > 0.7: σ ← min(σ × 1.25, σ_max=1.5 m)
- If pass-rate < 0.3: σ ← max(σ × 0.8, σ_min=0.05 m)

Distributed initialization enabled (same as IROS21 paper).

**Pre-validation:** run a 5M-step pilot with one seed before launching the full 100M × 3 seeds, to confirm σ adapts in the right direction and learning is non-degenerate. (Risk register §9.)

---

## 4. Held Constants (Critical)

These values come from the Optuna TPE sweep (Stage 2, 50M from D_bent_3_30 → 7/8 det gates, 10/10 episodes, R=89.66). The sweep cleanly outperformed the manual HPs by 3.5× at the same compute budget. They are **locked across all conditions and all rungs**.

| Element | Value | Source / Why |
|---|---|---|
| `learning_rate` | **3e-4** | Sweep winner (trial 26) |
| `n_steps` (per env) | **500** | Sweep winner — TPE converged on small rollouts |
| `n_epochs` | **10** | Sweep winner |
| `batch_size` | **50000** | = `n_steps × n_envs` = 500 × 100 (full-batch update) |
| `gamma` | **0.9794** | Sweep winner |
| `gae_lambda` | **0.9635** | Sweep winner |
| `clip_range` | **0.3** | Sweep winner — TPE moved AWAY from 0.2 default |
| `ent_coef` | **0.05** | **Sweep winner — KEY surprise.** TPE moved to high entropy to break past gate 1. Do not compress mid-experiment. |
| `vf_coef` | **0.5** | Sweep winner |
| `max_grad_norm` | **0.5** | Sweep winner |
| `net_arch` | **MLP [256, 256]** | Pinned (sweep's net_arch axis was effectively unsampled because warm-start arch is baked in) |
| `activation_fn` | **Tanh** | Same reason |
| Parallel envs | 100 | Same throughput across conditions |
| Total env steps per condition | **100M** | Equal compute across A/B/C |
| `body_rate_anneal_steps` | 25M | Locked from Bug 10 fix |
| `deviation_penalty_coeff` | 0.01 | Locked from Bug 10 fix |
| `spline_deviation_ratio` | 1.5 | Locked from Bug H fix |
| Reward function | spline progress + deviation + body-rate (no gate bonus, no crash penalty) | Paper-faithful |
| Observation space | 26-dim IROS21 | Locked |
| Action space | 4-dim per-motor thrust | Locked |
| `gate_width` | 5 m | Locked |
| `distributed_init` | true (training only) | Paper-faithful; eval forces `start_pos` via `set_random_reset(False)` |
| VecNormalize | enabled, stats saved per condition | Same protocol |
| Evaluation | deterministic, 20 episodes per track | Same protocol |
| Seeds | **3 per condition (9 total runs)** | Minimum for error bars |

If anything in this table changes mid-experiment, the paper has to disclose it. **Lock it now.**

> **Note on `ent_coef=0.05` vs prior overnight memory:** The auto-memory note "bump ent 0.01→0.02 to break plateaus, drop back to 0.01 if 0.03 stops helping" was correct *for the prior manual chain*. The Optuna sweep tested up to 0.05 and TPE converged there for the warm-started circle_8 fine-tune. For this experiment, follow the sweep result, not the older heuristic.

---

## 5. Metrics

### Primary (reported as mean ± std across 3 seeds)

| Metric | Description |
|---|---|
| **Final gate-pass rate** on target track (`circle_8`) | Deterministic, 20 eps, mean gates / 8 |
| **Sample efficiency** | Env steps to first 8/8 deterministic eval, or "did not reach" |
| **Zero-shot generalization** | Mean gate-pass rate on held-out tracks (see §7) without fine-tuning |
| **Robustness to gate perturbations** | Gate-pass rate when held-out track gates are perturbed by `N(0, 0.5²)` m |

### Secondary (for analysis sections)

- Training reward curve (`ep_rew_mean` vs env steps), with rung boundaries marked for B
- Entropy curve (`policy_entropy` vs env steps) — diagnoses commit
- Episode length curve — diagnoses hover-collapse failure mode
- Wall-clock training time per condition
- Action L2 norm at evaluation — diagnoses smoothness

### Diagnostic (only if something goes wrong)

- Failure-mode classification per eval rollout: `cleared / hover-collapse / dive-and-crash / late-altitude-collapse`
- Saved per-rollout state trajectories for animation in the paper

### Statistical analysis protocol

3 seeds gives weak power for traditional t-tests. Report instead:

- **Per-condition CI:** 95% bootstrap CI over the **3 seeds × 20 deterministic episodes = 60 episodes** per condition-track pair. The unit of analysis is the episode (gates passed), so n=60 gives reasonable CI width even though the seed n=3.
- **Cross-condition contrast:** for the headline numbers (A vs B, B vs C), report the difference in mean gates with a 95% bootstrap CI over the per-episode pairings. Treat any non-overlap with 0 as the effect-size claim. Do *not* claim p-values from a 3-seed t-test.
- **Effect-size disclosure:** state seed-level mean and std explicitly in the results table (e.g., "B: 7.2 ± 0.4 gates across seeds {7.0, 7.5, 7.0}"). Reviewers will compute their own gut-feel from this.
- **Same eval seed across conditions:** fix the eval-rollout seed (env start randomization, action sampling) so all 9 final checkpoints are evaluated on the same 20 trajectories per track. This is what makes the per-episode bootstrap valid.

For a class project / workshop, this is honest. For a venue paper (IROS, CoRL, RA-L), bump to 5 seeds — the marginal compute (~17 hr more on the VM) buys substantially tighter CIs.

---

## 6. Compute Budget

Training will run on **GCP Compute Engine `c3-standard-8`** (8 dedicated vCPU, 32 GB RAM, latest Intel Xeon, ~$0.42/hr on-demand) — see §13 for full cloud setup spec. The local UTM ARM64 VM is retained for development/smoke tests only; final-results runs all happen on GCP for reproducibility and isolation.

**Throughput projection:** local UTM VM measured ~10K FPS at 100 envs. Dedicated 8-core x86 with no hypervisor contention should hit ~40–60K FPS at 100 envs (4–6× speedup). Plan around a **conservative 40K FPS** assumption to leave margin; if it runs faster, that's just headroom.

| Condition | Steps | Wall-clock per seed @ 40K FPS | × 3 seeds |
|---|---|---|---|
| A — No curriculum | 100M | ~42 min | ~2.1 hr |
| B — Hand-designed (6-rung, 10+10+12+13+15+40) | 100M | ~42 min | ~2.1 hr |
| C — Automated random-track | 100M | ~42 min | ~2.1 hr |

**Training subtotal: ~6.3 hr.** Add eval grid (~1 hr for 9 checkpoints × 7 tracks) + Condition C pilot (~5 min for 5M steps) + smoke tests + buffer for reruns. **Realistic total: 9–12 hr of GCP wall-clock**, fitting in a single working day if started early.

**Cost projection:**

| Phase | Hours | Cost on `c3-standard-8` @ $0.42/hr |
|---|---|---|
| First-time x86 build + smoke tests | ~2 hr | $0.85 |
| Condition C pilot (5M steps + sampler validation) | ~0.5 hr | $0.21 |
| Full 9-run training campaign | ~6.3 hr | $2.65 |
| Eval grid (9 ckpt × 7 track) | ~1 hr | $0.42 |
| Buffer for reruns / debugging | ~3 hr | $1.26 |
| **Total** | **~13 hr** | **~$5.40** |

Budget assumes one boot-disk snapshot (~$0.10/month for 30 GB) and outbound egress for `rsync`-back of `saved/` (~few GB, ≪$1). **Comfortable headroom on the $50 GCP class credit** — projected use ~$10–15 including any iteration on the random-track sampler.

This is dramatically less than the existing `curriculum_v2/` work (~230M+ steps) because it locks hyperparameters and doesn't iterate per-rung.

**Caveat:** if Condition B reaches `bent_3_30` (rung D) with the locked HPs and ep_rew_mean is still climbing at the 13M mark, *do not* extend the rung — that contaminates the equal-budget comparison. Take the result as-is and let the next rung pick up. The 6-rung budget split is part of the experimental design, not a tunable.

---

## 7. Evaluation Protocol

### Target track (in-distribution)
- `circle_8` — primary number reported in the abstract
- Deterministic policy, 20 episodes, mean ± std gates passed
- Episode terminates at ground crash, OOB, or 30s timeout (no gate-collision termination)
- Eval forced to start from `start_pos` (`set_random_reset(False)` per Bug I)

### Held-out tracks (generalization)

Picked to **not** appear in Condition B's ladder (straight_2, straight_3, bent_3_15, bent_3_30, bent_3_45, circle_8):

- `oval_8` — closed loop with varying curvature; analogous topology to circle_8 but distinct geometry
- `helix_8` — adds altitude variation (vertical axis stress test)
- `vertical_3` — short track with strong altitude component (vertical generalization)

If the stretch rung G (figure_8) was *not* run for Condition B, also include `figure_8` as a held-out for the in-plane direction-reversal stress test. (If figure_8 *was* run as a B rung, drop it from held-out.)

Same eval protocol (20 deterministic eps, `set_random_reset(False)`). Reported as mean across the held-out set.

### Robustness
Apply gate perturbation `N(0, 0.5²)` m to each held-out track. 20 episodes per track. Reports robustness gap (zero-shot generalization gates − robustness gates).

---

## 8. Implementation Checklist

Ordered by priority. Stop at any line and the experiment is still partially salvageable.

**Local-machine prep (do before paying for GCP minutes):**

1. [ ] Lock hyperparameters in `experiment_hparams.json` (the §4 table, ready for `_train_one_hp.py`'s JSON-driven loader). Commit.
2. [ ] Build `run_condition_A.sh` — single 100M call to `_train_one_hp.py` on `circle_8`, parameterized by `--seed`.
3. [ ] Build `run_condition_B.sh` — 6 sequential rungs, calls `track_loader.py apply` between each, hands off `--load-model` + `--load-vecnorm`, parameterized by `--seed`.
4. [ ] Smoke-test B locally with a tiny budget (e.g., 100K per rung, ~2 min total) to catch wiring bugs before paying for GCP.
5. [ ] Add a `--seed` flag to `gate_racing_iros.py` (currently only `_train_one_hp.py` has it). Propagates to PPO + numpy + env C++ RNG. *Or* commit to using `_train_one_hp.py` everywhere and skip this.
6. [ ] Implement Condition C: per-episode random-track sampler with adaptive σ callback (§3). Add a unit test that confirms σ adapts in the right direction on synthetic success rates (§9 mitigation).

**Cloud migration (one-time, ~2 hr):**

7. [ ] Push repo to private GitHub (or GitLab) with `saved/` and `logs/` in `.gitignore`. Tag the commit you intend to run from (e.g., `experiment-v1`).
8. [ ] Provision GCP `c3-standard-8` per §13. Use the boot-disk image specified there.
9. [ ] Run `cloud_setup.sh` (build flightmare on x86 from a fresh clone). Verify `python diag.py` passes.
10. [ ] Snapshot the boot disk once everything builds — future runs spin up from this image without rebuilding.
11. [ ] **Pin reproducibility artifacts:** record git SHA, `flightgym.so` SHA-256, and `conda env export > experiment_env.yaml` into a `RUN_MANIFEST.md` in the repo. All 9 runs use these *exact* artifacts; no rebuilds or env updates mid-experiment.
12. [ ] Cloud smoke test: same tiny-budget B run from step 4, on the GCP box, to confirm parity with local results.

**Run experiments (on GCP):**

13. [ ] Pilot Condition C: 5M steps, 1 seed → check σ adapts in expected direction, gates start being cleared by ~3M.
14. [ ] Run Condition A × 3 seeds (seeds 0, 1, 2). Bank `ep_rew_mean` curve, final eval, VecNormalize stats. Run inside `tmux` so SSH disconnects don't kill the job.
15. [ ] Run Condition B × 3 seeds (seeds 0, 1, 2 reused for fair comparison). Bank per-rung checkpoints + final.
16. [ ] Run Condition C × 3 seeds. Bank.
17. [ ] Run eval grid on all 9 final checkpoints across target + 3-or-4 held-out + same set perturbed. **Use a fixed eval-rollout seed across all conditions** so per-episode bootstrap CIs (§5) are valid.

**Wrap-up:**

18. [ ] `rsync` `saved/` and `logs/` back to local Mac. **Stop the GCP instance** (don't delete — keep it stopped for cheap re-spin if reruns are needed). After the paper deadline, delete the instance and its boot snapshot.
19. [ ] Generate plots (reward curve overlay with rung boundaries, generalization bar chart with bootstrap CIs, robustness bar chart, failure-mode breakdown).
20. [ ] Write the paper.

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Condition C (automated curriculum) fails to converge | Medium | Pre-validate with the 5M-step pilot before committing 75M × 3 seeds |
| 100M steps insufficient for B to reach circle_8 | Low | If pilot seed only reaches 4/8 on circle_8 rung F by 25M (well before the 40M cap), the curriculum has a real problem; investigate before extending. Do not silently increase budget — that contaminates the equal-compute comparison. |
| 3 seeds yields noisy error bars | Medium | Use per-episode bootstrap CIs (§5 statistical analysis) rather than seed-level t-tests. Report seed-level means explicitly. If budget allows post-experiment, add 2 more seeds (≈17 hr). |
| Existing `curriculum_v2/` checkpoints tempt reuse | High (see §10) | Don't. Lock the hyperparameters first; rerun cleanly |
| Random-track sampler implementation bug | Medium | Add a unit test that confirms σ adapts in the right direction on synthetic success rates |
| `ent_coef=0.05` over-explores at later rungs | Medium | Locked by design — this is what the sweep found. Do not compress mid-run; that re-introduces the v1 problem. If circle_8 plateaus < 5/8 across all 3 seeds, flag in limitations and discuss in paper as a finding. |
| Distributed init Phase 2 doesn't engage on a rung | Medium | If `pass_rate < 0.3` for 5M steps, log a warning. Still report the result; do not intervene. |
| Stretch figure_8 rung consumes budget needed for circle_8 | High if attempted | Do not attempt unless circle_8 rung F finishes at ≥7/8 with steps to spare. |

---

## 10. On Reusing Existing Curriculum Results

**Recommendation: do not reuse. Rerun Condition B from scratch with locked hyperparameters.**

The existing runs in `saved/curriculum_v2/E_circle_8_v4/` (the 10/8 winner) and `saved/sweep_stage2_trial26_50M/` (7/8) are engineering exploration, not controlled experimental data. Using them in the paper would create the following problems, any one of which is fatal to the comparison:

| Problem | Evidence |
|---|---|
| Hyperparameters varied across rungs in `curriculum_v2/` | `bent_3_30` used `ent_coef=0.003` after exploration kick; `circle_8` v3–v6 used `ent_coef ∈ {0.05, 0.001, 0.0003, 0.0001, 0.00005}`; LR was reduced for fine-tunes |
| Single seed everywhere | No error bars; reviewer/grader question #1 will be "is this seed noise?" |
| Mid-run policy / env changes | `body_rate_anneal_steps` switched 5M → 25M mid-experiment; `deviation_penalty_coeff` from 0 → 0.01; `gate_bonus` toggled 0 → 5 → 10 → 0 across the figure_8 chain |
| Step counts inconsistent | rung 4 took 60M; rung 5 took 25M from rung 4; circle_8 cumulative count not tracked across v1→v4 |
| Eval protocol drift | Different episode counts (10 vs 20), different `min-gates` thresholds |
| `sweep_stage2_trial26_50M` used a different start (D_bent_3_30, not target) | Cannot be cleanly attributed to either A or B condition |

If you reused these, you would not be able to honestly answer:
- "How many env steps did Condition B take?" — answer is undefined.
- "Are differences between B and C statistically significant?" — answer is unknowable with 1 seed.
- "What was the hyperparameter setting?" — answer is "it varied."

These are the first three questions any reader will ask.

**What you can keep from the existing work:**
- The track YAMLs themselves (`flightlib/configs/tracks/*.yaml`) — those are inputs, not data.
- The reward function, env config, build artifacts.
- The bug fixes from `FIXES_ATTEMPTED.md`. Those define the env you're running.
- The qualitative knowledge of which rungs are hard — useful for the 6-rung ladder choice in §3.
- The sweep `best_hparams.json` — this is the *input* to the new experiment, not data from it.

**What you should not keep:**
- Any policy checkpoint, reward curve, eval result, or VecNormalize stats from prior runs.
- Treat `saved/curriculum_v2/` and `saved/sweep_stage2_trial26_50M/` as background — interesting context, not paper data.

The good news: a clean rerun of Condition B at locked hyperparameters should be ~6.3 hrs of compute (3 seeds × 2.1 hr), not the weeks the original took. The original was slow because it was iterative; the rerun is one-shot.

---

## 11. Open Questions (decide before launching)

1. ~~Reduced 7-rung ladder: which 7?~~ **Resolved.** 6-rung ladder: `straight_2 → straight_3 → bent_3_15 → bent_3_30 → bent_3_45 → circle_8`. Rationale: each hop has been empirically traversable in `curriculum_v2/`; `bent_3_45` is added to bridge the bent_3_30 → circle_8 gap that needed 50M in the prior sweep run.
2. ~~Held-out track choice: confirm none overlaps with the ladder.~~ **Resolved.** Held-out: `oval_8, helix_8, vertical_3` (+ `figure_8` if not run as stretch rung). All disjoint from the ladder.
3. **ACL adaptive σ schedule:** pilot to find the right thresholds. Current proposal: σ_init=0.2 m, ×1.25 above 0.7 success, ×0.8 below 0.3, capped at 1.5 m. Decide after the 5M pilot whether to tighten.
4. **Distributed init in Condition A:** enabled (paper-faithful) — failure is attributable to lack of curriculum, not to init choice. Locked.
5. **Figure_8 stretch rung:** attempt only if circle_8 rung F finishes at ≥7/8 across all 3 seeds with budget to spare. Otherwise skip and discuss as a known limitation in §12.
6. **Random-track topology base:** `circle_8` (target track). Alternative would be a generic 8-gate closed loop sampled topology-up; the simpler choice (perturb circle_8) makes A/C comparison cleanest. Locked unless pilot reveals collapse.

---

## 13. Compute Environment (GCP)

### Instance specification

| Setting | Value | Why |
|---|---|---|
| Provider | Google Cloud Platform Compute Engine | $50 class credit available; using class-issued credentials |
| Region | `us-central1` (Iowa) — fallback `us-east1` | c3 family widely available; lowest latency from continental US |
| Machine type | **`c3-standard-8`** | 8 vCPU dedicated, 32 GB RAM, latest Intel Sapphire Rapids Xeon. CPU-optimized → ~10–20% faster physics sim per core than `n2`. Plenty of RAM (vs 1.9 GB OOM ceiling on local UTM VM) |
| OS image | Ubuntu 22.04 LTS x86_64 (`ubuntu-2204-jammy-v*`) | Closest match to local Ubuntu 24.04 ARM64; well-tested with conda-forge / pybind11 / yaml-cpp toolchain |
| Boot disk | 30 GB Standard Persistent Disk (pd-standard) | Holds repo + conda env + saved/ + logs/ for all 9 runs (~10 GB total). Standard PD is fine — IOPS not the bottleneck for this workload |
| Pricing model | **On-demand** | Spot is ~$0.10/hr cheaper but adds preemption-handling complexity not worth saving $7 on a $50 budget; on-demand also avoids any chance of mid-run interruption that could contaminate the equal-step comparison |
| Network egress | ~few GB out at end (`rsync` saved/ back) | < $1 of egress; negligible |

**Why not GPU?** Workload is CPU-bound: flightmare's C++ physics sim is the throughput bottleneck, not the MLP[256,256] forward pass. A T4 or A10 sits idle while waiting for the env. Verified empirically by the existing CPU-only training pipeline (`device="cpu"` in `_train_one_hp.py`).

**Why not preemptible/spot?** Checkpoints save every 500K steps so technically recoverable, but mid-rung preemption resets the rollout buffer and rolling stats — this could subtly bias one seed's curve and break the equal-budget guarantee §4 relies on. On-demand removes that risk for ~$7.

### Cloud setup workflow (one-time)

```bash
# 1. Provision via gcloud (or GCP console GUI, equivalent)
gcloud compute instances create flightmare-train \
  --zone=us-central1-a \
  --machine-type=c3-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard

# 2. SSH in
gcloud compute ssh flightmare-train --zone=us-central1-a

# 3. On the box: install miniforge, clone, build (one-time, ~30 min)
bash <(curl -fsSL https://raw.githubusercontent.com/conda-forge/miniforge/main/Miniforge3-Linux-x86_64.sh)
source ~/miniforge3/etc/profile.d/conda.sh
git clone <YOUR_PRIVATE_REPO_URL> ~/flightmare
cd ~/flightmare
bash cloud_setup.sh   # to be written — see Implementation Checklist step 9

# 4. Verify build
cd ~/flightmare/flightrl/training
conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare
python diag.py        # should print obs dims, reward range, FPS

# 5. Snapshot the boot disk (one-time, after build verified)
gcloud compute disks snapshot flightmare-train --snapshot-names=flightmare-built-v1 \
  --zone=us-central1-a
```

After the snapshot, future instances boot from `flightmare-built-v1` and skip the build entirely.

### Running experiments (on the cloud box)

```bash
# In tmux so SSH disconnects don't kill it
tmux new -s train

# Launch the full campaign
cd ~/flightmare/flightrl/training
conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare

bash run_all_experiments.sh   # to be written — calls A/B/C × seeds {0,1,2} sequentially

# Detach: Ctrl-b d. SSH out. Reconnect later: gcloud compute ssh ... -- -t tmux attach -t train
```

### Cost discipline

- **Stop, don't delete** the instance between sessions: stopped instances bill only for storage (~$0.04/day for 30 GB pd-standard) — pennies — but spin back up in seconds.
- **Don't leave it running idle.** GCP bills per second while running. If you're not actively iterating, stop it.
- **Monitor credit burn** at https://console.cloud.google.com/billing — set a $30 budget alert (60% of credit) so you notice before exhaustion.
- **`rsync` results back** at the end so you can delete the instance after the paper deadline:
  ```bash
  # On local Mac
  gcloud compute scp --recurse flightmare-train:~/flightmare/flightrl/training/saved/ ./saved_gcp/ \
    --zone=us-central1-a
  ```

### Why not Colab Pro

- 24-hour runtime cap + idle disconnects → multi-day campaign requires reconnect choreography
- Notebook UI awkward for `tmux` + multi-stage curriculum scripts
- Free GPU is wasted on CPU-bound workload
- Storage via Google Drive is slow for frequent checkpoint writes
- Net: would cost more debugging hours than $10 of GCP credit

Colab Pro could be useful for **post-hoc plotting/analysis** if a local Jupyter on the Mac isn't preferred — but neither option requires it.

### Local UTM VM: development only

The existing local Ubuntu 24.04 ARM64 VM (`~/flightmare`) is retained for code development, smoke tests with tiny step budgets, and editing the orchestrator scripts. **No final-results runs happen locally.** This keeps results reproducible against a single (cloud) build artifact and avoids ARM-vs-x86 numerical drift contaminating the comparison.

---

## 14. What This Experiment Is NOT

- Not a faithful reproduction of Penicka et al. ICRA25's adaptive curriculum.
- Not a sim-to-real study.
- Not an observation-space ablation (the original PROJECT_CONTEXT.md goal — shelved or moved to a §X.Y subsection if compute permits).
- Not an action-space study (CTBR vs per-motor — out of scope, even though the prior figure_8 plateau analysis suggested CTBR might unlock it).
- Not a guarantee that figure_8 will be solved. The prior overnight chain showed a 6/8 structural wall that survived multiple interventions. Figure_8 results, if reported, are exploratory, not confirmatory.

Acknowledging the scope explicitly in the paper's Limitations section is good practice for a class project and prevents over-claiming.
