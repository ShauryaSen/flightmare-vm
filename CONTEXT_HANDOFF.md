# Context Handoff — Drone Racing Curriculum Training

Hand this to a new agent to pick up where we left off. Read `PROJECT_CONTEXT.md`, `TRACK_CURRICULUM_PLAN.md`, `FIXES_ATTEMPTED.md`, `flightrl/training/curriculum_log.md`, and `OVERNIGHT_PROGRESS.md` for the long-form details.

---

## What this project is

Faithful replication of the **IROS 2021 autonomous drone racing baseline** (Song et al.) in Flightmare using SB3 PPO. We're running a **curriculum** of progressively harder tracks (single gate → straight → bent → circle → figure_8 → alphapilot) to find the policy's failure points.

- VM: Ubuntu 24.04 ARM64, conda env `flightmare`
- Repo root: `~/flightmare`
- Always: `export FLIGHTMARE_PATH=~/flightmare` before any training command
- Train command pattern: `cd ~/flightmare/flightrl/training && python gate_racing_iros.py --train 1 ...`
- Track switching: `python track_loader.py apply <track_name>` mutates `flightlib/configs/quadrotor_env.yaml`. Read by C++ env at startup, no rebuild needed.

---

## Current best models

| Track | Path (under `flightrl/training/`) | Eval result |
|---|---|---|
| **bent_3_30** | `saved/curriculum/rung04_bent_3_30_v5/best_model.zip` | 3/3 gates, 20/20 ✅ |
| **bent_3_45** | `saved/curriculum/rung05_bent_3_45/best_model.zip` | 3/3 gates, 20/20 ✅ |
| **circle_8** | `saved/curriculum/rung06_circle_8_v4/best_model.zip` | 5/8 gates (clean half-circle, dives after G4) |
| **figure_8** | `saved/curriculum/rung12_figure_8/best_model.zip` | partial — training was killed at user request |

Each model dir has `best_model.zip`, `vec_normalize.pkl`, and `checkpoints/`. Always pair the model with its matching `vec_normalize.pkl` — eval without it produces nonsense.

The circle_8 v4 model is the **deepest curriculum-trained policy** (60M+ steps cumulative). It's strong on circle-like motion but **catastrophically forgot** straight flight: see `flightrl/training/viz/multi_track/summary.txt` — it gets 0/1 gates on `single_1` because it can't fly straight anymore.

---

## The recipe that worked (and where it stopped)

**Recipe (rungs 4 & 5, 30°/45° bent tracks):**
1. Continue PPO from previous rung's `best_model.zip` + `vec_normalize.pkl`.
2. Train **way longer** than the original plan budget — 20M+ per rung, not 4M.
3. **Lower `ent_coef`** from default 0.01 → 0.003 → 0.001 → 0.0003 over successive runs.
4. Lower entropy compresses policy std (e.g. 0.99 → 0.21), which closes the gap between the (working) stochastic policy and the (failing) deterministic eval.

**Where it stopped:** circle_8 plateaus at 5/8 deterministic gates even after 90M+ cumulative steps and ent_coef as low as 0.0001. Training reward keeps climbing to ~70 (≈87% of full lap) but deterministic eval doesn't improve past 5 gates. Drone clearly arcs through G0→G4 then loses altitude after G4 and ground-crashes.

**Tried and didn't help on circle_8:**
- High entropy (0.05) followed by low-entropy commit → regressed to 4/8 (worse local optimum).
- High entropy alone → ep_len grew (more fluttering past G4) but no commitment to a G4→G5 trajectory.

---

## What's currently running

User has their own training going at `~/flightmare-bc/`:
- `python gate_racing_iros.py --train 1 --load-model saved/bc_then_ppo_v2_peak/best_model.zip ...`
- `--save-path saved/bc_then_ppo_v3 --learning-rate 3e-5 --ent-coef 0.005 --total-timesteps 30000000`
- BC-warm-start + low-LR PPO fine-tune. ~848% CPU usage. **Don't compete for CPU.**

The figure_8 training I had running was killed mid-run at user request. Best checkpoint preserved at `saved/curriculum/rung12_figure_8/best_model.zip` — was at 7M new steps in, reward ~7 (low; would need much more training).

---

## Open problems / suggested next moves

1. **circle_8 5/8 ceiling** — drone dives after G4. Hypotheses: (a) policy can't generate enough thrust during the post-G4 turn to maintain altitude; (b) reward function has no strong altitude signal (paper-strict, no `floor_penalty`, no `crash_penalty`). Option C: re-add a small `floor_penalty: 1.0` and re-train. Requires C++ already supports it (see Bug M / Bug O in `FIXES_ATTEMPTED.md`) — coefficient currently 0.0 / removed; YAML field absent.

2. **Catastrophic forgetting** — circle_8 v4 can't fly straight anymore (0/1 on `single_1`, negative reward on bent_3_30). Multi-task training (sample tracks uniformly) or replay buffer would fix this. The C++ env supports per-instance track config via the YAML, so a Python-side scheduler that round-robins tracks during PPO rollouts could work without a rebuild.

3. **HP sweep is wired up but never launched.** Files in `flightrl/training/hp_sweep/`:
   - `_train_one_hp.py` — single-trial subprocess. Reads JSON of hyperparameters.
   - `hp_sweep_optuna.py` — Optuna TPE orchestrator, 11-D search space (per `param_sweep.md`).
   - `launch_hp_sweep.sh` — nohup launcher.
   - **Optuna isn't installed yet.** Run `pip install optuna` in the `flightmare` env before launching.
   - Default 10M steps × 24 trials × 8 hours.

4. **Figure_8 training is paused**, not failed. Best-so-far model is at `saved/curriculum/rung12_figure_8/`. To resume: apply track first (`python track_loader.py apply figure_8`), then load from that best_model.

---

## Useful commands

```bash
# Activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare
cd ~/flightmare/flightrl/training

# Apply a track (mutates flightlib/configs/quadrotor_env.yaml; no rebuild)
python track_loader.py apply circle_8

# Eval (deterministic, 20 episodes from start_pos)
bash curriculum_eval.sh saved/curriculum/rung06_circle_8_v4 8 16
# args: <run_dir> <min_gates_for_success> <required_count_out_of_20>

# Resume training from a checkpoint
nohup python -u gate_racing_iros.py --train 1 \
  --total-timesteps 30000000 \
  --load-model saved/curriculum/rung06_circle_8_v4/best_model.zip \
  --load-vecnorm saved/curriculum/rung06_circle_8_v4/vec_normalize.pkl \
  --save-path saved/curriculum/rung06_circle_8_v5 \
  --tb-log saved/tb_logs/curriculum/rung06_v5 \
  --ent-coef 0.001 \
  > logs/myrun.log 2>&1 &
# NOTE: SB3 PPO.load with --load-model does NOT reset num_timesteps,
# so --total-timesteps is the count of NEW steps to train, not the cumulative target.

# Visualize (matplotlib + animated GIF)
python visualize_policy.py \
  --model saved/curriculum/rung06_circle_8_v4/best_model.zip \
  --vecnorm saved/curriculum/rung06_circle_8_v4/vec_normalize.pkl \
  -n 5 --animate --animate-n 1 --speedup 1 \
  --out-dir viz/test
# add --stochastic for stochastic policy

# Multi-track eval (one stochastic GIF per track)
bash run_multi_track_eval.sh viz/multi_track

# TensorBoard
tensorboard --logdir saved/tb_logs --bind_all --port 6006
```

---

## Things to know that bit us

- **`PPO.load` keeps `num_timesteps` from the saved model.** With `reset_num_timesteps=False` (the default in our script when `--load-model` is set), `model.learn(total_timesteps=N)` trains for `N` *additional* steps, but the TB and progress bar show the cumulative count. Easy to read wrong.
- **`TrainingRewardCallback`** in `gate_racing_iros.py` saves `best_model.zip` whenever the rolling 100-ep mean reward exceeds the prior best (gated by `min_steps_before_save=300_000` so a barely-warm policy can't clobber a real one — Bug J in `FIXES_ATTEMPTED.md`). Per-run `--save-path` keeps each curriculum rung isolated.
- **Eval (`gate_racing_iros.py --train 0`) calls `set_random_reset(False)`** so all envs start at `start_pos`. Required for honest deterministic eval — see Bug I in `FIXES_ATTEMPTED.md` for the bug where eval was masking distributed-init.
- **The `.so` import order trap**: `flightrl/training/flightgym.cpython-38-aarch64-linux-gnu.so` takes precedence over the freshly built `flightlib/flightgym.*.so`. After any C++ change, copy the rebuilt `.so` from `flightlib/` over the one in `flightrl/training/`. Otherwise new pybind methods raise `AttributeError`.
- **`-Ofast` breaks `std::isfinite(NaN)`** — fixed via volatile-based check in `command.cpp` (Bug 1).
- Reward is paper-strict right now: no `crash_penalty`, no `survival_bonus`, no `gate_bonus`. Just `r_progress + r_deviation - body_rate_penalty`. See PROJECT_CONTEXT.md §3 for the full spec.

---

## Files most likely to matter

- **`PROJECT_CONTEXT.md`** — full project / env / paper spec.
- **`TRACK_CURRICULUM_PLAN.md`** — original 13-rung curriculum design.
- **`FIXES_ATTEMPTED.md`** — append-only debug log. Read before proposing any fix.
- **`flightrl/training/curriculum_log.md`** — original sweep results (pre-extension).
- **`OVERNIGHT_PROGRESS.md`** — most recent run log (this session).
- **`flightrl/training/gate_racing_iros.py`** — main train + eval script.
- **`flightrl/training/visualize_policy.py`** — matplotlib + GIF visualization.
- **`flightrl/training/track_loader.py`** — track YAML switcher.
- **`flightlib/configs/quadrotor_env.yaml`** — env config (track + reward + dynamics).
- **`flightlib/configs/tracks/`** — per-track YAML library (`single_1`, `straight_3`, `bent_3_*`, `circle_*`, `oval_8`, `helix_8`, `vertical_3`, `figure_8`, `alphapilot`, plus a few legacy formats).
- **`flightrl/training/hp_sweep/`** — Optuna sweep code (untested; needs `pip install optuna`).
