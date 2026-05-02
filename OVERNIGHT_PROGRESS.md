# Overnight Progress — 2026-05-02 → 2026-05-03

**Goal (from user):** Get the drone flying through circle_8 by any means necessary; if successful, visualize circle_8 and figure_8 and continue the curriculum. If circle_8 looks structurally hopeless, fall back to the PPO hyperparameter sweep in `param_sweep.md`. **Never stop trying.**

This doc is append-only. Most recent attempt is at the bottom.

---

## Curriculum status entering this run

| # | Track | Outcome | Notes |
|---|---|---|---|
| 0 | single_1 | ✅ | 20/20 |
| 1 | straight_2 | ✅ | 20/20 |
| 2 | straight_3 | ✅ | 20/20 |
| 3 | bent_3_15 | ✅ | 20/20 |
| 4 | bent_3_30 | ✅ | **20/20 after 60M steps + ent_coef=0.003** (was previously called "failure rung" at 8M; deterministic-eval ceiling was just compressed-std issue) |
| 5 | bent_3_45 | ✅ | 20/20 after 25M from rung 4 (same recipe) |
| 6 | circle_8 | ⚠️ in progress | **focus of overnight run.** v2 (30M from rung 5, ent_coef=0.003): training reward climbed 9→42.5, but deterministic eval still 1/8. v3 (30M, ent_coef=0.001) running now. |
| 10 | vertical_3 | ✅ | (done previously in original sweep) |

**Recipe that worked for rungs 4 & 5:** more iterations + reduce ent_coef to compress policy std, bringing deterministic action close to the (already-working) stochastic mean.

---

## Plan for tonight

1. **Drive circle_8 to completion** (this is the load-bearing problem).
   - First: let the in-flight v3 (ent_coef=0.001, 30M) finish, eval, visualize.
   - If still failing: try **higher entropy** (per user request) — fresh-start the policy from rung 5 with ent_coef=0.05 to break out of the local "go straight" minimum.
   - If still failing: try **slow-and-steady** — lower the demanded speed by simulating slower commitment via lower learning rate + gate_bonus reintroduction.
   - If circle_8 looks structurally impossible after multiple attempts → switch to the **hyperparameter sweep** in `param_sweep.md` and let it run overnight.
2. **Visualize circle_8 and figure_8** as soon as either passes.
3. **Continue curriculum** through rung 7+ if circle_8 clears.

---

## Run log

### 04:50 UTC — Rung 6 v3 launched

- Loaded from rung 6 v2 (which itself was 30M from rung 5 with ent_coef=0.003).
- ent_coef dropped to 0.001 to keep compressing policy std (currently 0.41).
- Total timesteps: 30M new (cumulative will hit ~162M when done).
- Save dir: `saved/curriculum/rung06_circle_8_v3/`.
- Log: `logs/curriculum_rung06_v4.log`.

Reward trajectory so far (sampled): 46.7 → 47.4 → 48.7. Was 42.5 at v2 cutoff. Climbing.

### 04:55 UTC — HP sweep infrastructure prepped

In case circle_8 turns out to be structurally hopeless, the sweep is ready to launch:
- `flightrl/training/hp_sweep/_train_one_hp.py` — single-trial training subprocess
- `flightrl/training/hp_sweep/hp_sweep_optuna.py` — Optuna TPE orchestrator with 11-D search space (per `param_sweep.md`)
- `flightrl/training/hp_sweep/launch_hp_sweep.sh` — nohup launcher

Optuna isn't installed yet; will `pip install optuna` only if the sweep is actually needed.

### 14:03 UTC — Rung 6 v3 finished — **major breakthrough on circle_8**

Eval result on `saved/curriculum/rung06_circle_8_v3/best_model.zip`:

| Metric | v2 (ent=0.003) | **v3 (ent=0.001)** |
|---|---|---|
| gates_passed (deterministic, 20 eps) | 1.00 ± 0.00 | **5.00 ± 0.00** |
| Mean reward (deterministic) | 12.42 | **58.36** |
| ep_len_mean | 120 | 189 |
| Policy std at end of training | 0.412 | **0.288** |

5/8 gates in deterministic mode — drone now flies the **entire top half** of the circle cleanly through G0→G1→G2→G3→G4, then loses altitude after G4 and ground-crashes around (-24, -8). Visualization at `flightrl/training/viz/curriculum/best_model_20260502_140336/best_model_topdown.png`.

Hypothesis confirmed: the recipe (lower ent_coef + more iterations) breaks the sustained-turning ceiling. The "circle_8 fail" verdict from the original 8M-step sweep was the same under-trained-deterministic-policy story as bent_3_30.

### 14:03 UTC — Rung 6 v4 launched

- Loaded from v3.
- **ent_coef=0.0003** (3× smaller still) to compress std further (0.288 → ?).
- 30M new steps, save dir `saved/curriculum/rung06_circle_8_v4/`.
- Goal: push from 5/8 → 8/8 gates.

### 14:50 UTC — Rung 6 v4 result + new plan

V4 finished. Eval still **5/8 gates** but training reward climbed (55.6 → 64.3) and ep_len 189 → 203, so the policy IS still improving — just not enough to push deterministic past G4.

| Metric | v3 | **v4 (ent=0.0003)** |
|---|---|---|
| Eval gates | 5 | 5 |
| Eval reward | 58.36 | 60.66 |
| Eval ep_len | 189 | 203 |
| Train ep_rew_mean (final) | ~48 | ~64 |
| Policy std | 0.288 | 0.208 |

Visualization at `viz/curriculum/best_model_20260502_145136/` shows drone diving (z: 3 → 0) immediately after G4 — failure mode is altitude collapse, not a yaw/bank issue. Compressing std further is hitting diminishing returns.

**New plan: high-entropy exploration phase** to discover a non-diving G4→G5 strategy.

### 14:51 UTC — Rung 6 v5 (high entropy) launched

- Loaded from rung 6 v4 best.
- **ent_coef=0.05** (50× higher!). Goal: kick the policy out of the "dive after G4" basin.
- 20M steps to encourage exploration; will follow with low-entropy commit run if eval improves.
- Save dir `saved/curriculum/rung06_circle_8_v5_highent/`. Log `logs/curriculum_rung06_v6_highent.log`.

### 15:25 UTC — V5 high-entropy result + v6 commit launched

V5 finished. Eval still **5/8 gates** but ep_len jumped to **231** (up from 203) and std doubled (0.208 → 0.413), meaning the drone is now fluttering past G4 for longer before crashing. The high entropy didn't discover a *new* strategy — it just made the existing strategy noisier. Reward dropped slightly (60.7 → 56.5) as expected.

But ep_len growth is a real signal: the policy *can* go past G4, just needs commitment to a stable G4→G5 strategy.

Launching **v6 commit run** with **ent_coef=0.0001** (lowest yet) for 30M steps. Goal: collapse the post-G4 fluttering into a single committed trajectory through G5.

| Run | ent_coef | Eval gates | Eval R | Eval ep_len | Std |
|---|---|---|---|---|---|
| v3 | 0.001 | 5 | 58.4 | 189 | 0.288 |
| v4 | 0.0003 | 5 | 60.7 | 203 | 0.208 |
| v5_highent | 0.05 | 5 | 56.5 | 231 | 0.413 |
| **v6_commit** | 0.0001 | 4 | 44.6 | 151 | 0.167 |

### 16:30 UTC — V6 commit regressed; pivoting to figure_8 per user

V6 commit (low-ent from v5 highent) **regressed** to 4/8 (worse than v4's 5/8). The high-ent → low-ent path landed in a worse local optimum than the steady low-ent path. Best circle_8 deterministic checkpoint remains **v4 at 5/8 gates / R=60.7 / ep_len=203** — drone cleanly arcs the top half of the circle but dives after G4.

User direction shift: **try figure_8 instead** to see if the policy transfers better there. (figure_8 alternates curvature direction at the origin crossover — fundamentally different from circle_8's monotonic turning.)

### 16:35 UTC — Figure_8 training launched (rung 12)

- Track applied: `figure_8` (8 gates, 6m radius lobes, z=3, start (0,-4,3) facing +y).
- Loaded from circle_8 v4 best (the deepest learned policy on a sustained-turning track).
- ent_coef=0.003 (entropy bumped vs the v6 commit value, since the policy needs to explore the curvature reversal, not just refine).
- 30M steps. Save dir `saved/curriculum/rung12_figure_8/`.
- Initial reward 0.6 (very low — circle_8 policy doesn't know how to handle the direction reversal). ep_len 99.

Going to let this run and see what figure_8 looks like with policy transfer.
