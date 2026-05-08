# Overnight Plan — 2026-05-03 → 2026-05-04

**Operator:** Claude (autonomous, looping)
**Owner reviews in the morning:** Shaurya
**Live progress file:** [OVERNIGHT_PROGRESS.md](OVERNIGHT_PROGRESS.md)

## Context at handoff

- bent_3_30 fine-tune from C_bent_3_15 currently running, PID 36084. Started ~27.3M, target 52.3M total, currently iteration 56 / 52.2M, ep_rew_mean ≈ 22, ep_len ≈ 265, std=1.39. Best saved at saved/curriculum_v2/D_bent_3_30/best_model.zip (R=22.04). Should finish in ~1 iteration.
- Curriculum-v2 saves: `saved/curriculum_v2/{A_straight_2, B_straight_3, C_bent_3_15, D_bent_3_30}/`. Older curriculum/ dir has previous attempts.
- Memory says: budgets ≥10M each, bent/circle 25M+, circle up to 50M; never declare failure while reward climbing. Entropy: bump 0.01→0.02 to break plateaus, drop back to 0.01 if 0.03 stops helping.
- Distributed init currently `false` in YAML. Curriculum has been running with no-distributed-init (always start from start_pos).

## Goals (in priority order)

1. **Circle_8 — all 8 gates passed deterministically.** Train 50M+ steps from D_bent_3_30. Iterate as needed until success. Visualize.
2. **Figure_8 — all 8 gates passed.** Train from circle_8. Visualize.
3. **PPO hyperparameter sweep.** Identify best config.
4. **Final policy push using sweep results.**

## Hard rules

- Never declare a rung failed while ep_rew_mean is still trending up.
- Use ent_coef=0.01 by default. Bump to 0.02 only on clear plateau (no improvement for ≥10M steps with reward flat). Drop to 0.01 if 0.03 hurts.
- Always save with `--save-path` to a unique dir (avoid clobbering — see Bug J 2026-05-01).
- Always carry forward both `--load-model` AND `--load-vecnorm`.
- After every meaningful step, append a row to OVERNIGHT_PROGRESS.md.
- Never bypass `track_loader.py apply <name>` before changing tracks — it cycles start_pos + gates in `flightlib/configs/quadrotor_env.yaml`.
- Always check `ps aux | grep gate_racing` before launching new training. Only one training run at a time on this VM (8 cores).

## Sequence

### Phase 1 — finish + eval bent_3_30 (~5 min)
- Wait for PID 36084 to exit.
- Eval: `python gate_racing_iros.py --train 0 --model-path saved/curriculum_v2/D_bent_3_30/best_model.zip --vecnorm-path saved/curriculum_v2/D_bent_3_30/vec_normalize.pkl --n-eval-episodes 20 --min-gates 3 > logs/eval_bent_3_30.log 2>&1`
- Record gates_passed and lap_completed in progress doc.

### Phase 2 — circle_8 attempt 1 (50M steps, ~70 min)
- `python track_loader.py apply circle_8`
- Train from D_bent_3_30 with 50M total_timesteps, ent_coef=0.01, save to `saved/curriculum_v2/E_circle_8/`, tb-log `saved/tb_logs/curriculum_v2_E`.
- Run with `nohup ... &` and capture PID.
- Loop check: every iteration, monitor `tail` of log. Track ep_rew_mean trend (should monotonically rise as drone learns more gates).

### Phase 3 — circle_8 iteration if needed
If after 50M circle_8 deterministic eval is still <8/8:
- **Decision tree:**
  - If `ep_rew_mean` was still climbing at end → extend with another 25M from current best.
  - If reward plateaued AND `std` low (<1.0) → bump ent_coef to 0.02, fine-tune another 20M.
  - If reward plateaued AND `std` high (>1.5) → keep ent_coef=0.01, fine-tune another 20M.
  - If multiple ent attempts plateau at <4/8 → try BC warm-start: collect circle_8 demos via `trajectory_controller.py --speed 4.5 --save-path demos/circle_8_overnight.npz`, then `train_bc.py`, then PPO fine-tune at LR=3e-5.
  - If still stuck after BC → reduce gate_width experiment (1.2m) or apply small `gate_bonus=2`. Document in FIXES_ATTEMPTED.md before changing the YAML.

### Phase 4 — visualize circle_8 success
- `python visualize_policy.py --model saved/curriculum_v2/E_circle_8/best_model.zip --vecnorm saved/curriculum_v2/E_circle_8/vec_normalize.pkl -n 20 --animate --animate-n 3 --out-dir viz/E_circle_8`
- Confirm GIF + PNG produced. Note in progress doc.

### Phase 5 — figure_8 (50M+ from circle_8)
- `python track_loader.py apply figure_8`
- Train from E_circle_8 with 50M, save to `saved/curriculum_v2/F_figure_8/`.
- Same iteration logic as Phase 3.
- Visualize.

### Phase 6 — PPO hyperparameter sweep
After both circle_8 and figure_8 visualized successfully (or after 8 hours from now, whichever sooner — don't let the sweep get starved):
- Sweep on circle_8 (cheaper than figure_8, transfers well). Each condition gets ~3M steps from E_circle_8 best (or earlier good model). Eval at end.
- Conditions to sweep (8 conditions, ~3M each, ~25 min each, ~3.3 hours total):
  - Baseline: lr=3e-4, ent=0.01, batch=150K, n_epochs=10, clip=0.2 (already known)
  - lr=1e-4 (slower fine-tune)
  - lr=5e-4 (faster)
  - ent=0.005 (less exploration, more commitment)
  - ent=0.03 (more exploration)
  - clip=0.1 (more conservative updates)
  - n_epochs=20 (more updates per rollout)
  - gae_lambda=0.99 (longer horizon advantage)
- Each writes to `saved/sweep/<id>/`, deterministic eval logged.
- Pick best config by deterministic gates_passed + ep_rew_mean.

### Phase 7 — final push (use best sweep config)
- Train ≥30M with best config from sweep, on the hardest unsolved track.
- Visualize.

## Files & paths reference

- Tracks: `flightlib/configs/tracks/{single_1,straight_2,straight_3,bent_3_15,bent_3_30,bent_3_45,circle_8,circle_6,circle_4,oval_8,vertical_3,helix_8,figure_8,alphapilot}.yaml`
- Track loader: `flightrl/training/track_loader.py {list,show,apply,current} <name>`
- Train: `flightrl/training/gate_racing_iros.py`
- Visualize: `flightrl/training/visualize_policy.py`
- Demos collector (BC): `flightrl/training/trajectory_controller.py --speed N --save-path demos/X.npz`
- BC: `flightrl/training/train_bc.py --demos demos/X.npz [...] --save-dir saved/bc_X --epochs 200`
- Logs go in: `flightrl/training/logs/`
- TB logs: `flightrl/training/saved/tb_logs/curriculum_v2/`

## Reminders

- Each fine-tune launch: ent_coef passes through `--ent-coef`. Default 0.01 unless we have a clear reason.
- After C++ changes (none planned tonight): rebuild + copy .so to flightrl/training/. (Not needed for any of the planned phases.)
- Watchdog: every loop iteration, check that the training process is alive. If it died, capture last 200 lines of log into progress doc and decide whether to restart.
- Never run two trainings at once.
- TB: `tensorboard --logdir flightrl/training/saved/tb_logs/curriculum_v2 --bind_all --port 6006` (operator can attach in morning).
