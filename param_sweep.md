# PPO Hyperparameter Sweep — Implementation Plan

## Goal

Set up an automated **PPO hyperparameter sweep** using Optuna (TPE Bayesian optimization) to find the best-performing model configuration. This is distinct from the existing `sweep/` directory, which sweeps **reward shaping** parameters — this new sweep tunes the **PPO algorithm hyperparameters** themselves.

Inspired by the [Joel Baptista reference](https://joel-baptista.github.io/phd-weekly-report/posts/hyper-op/) which used W&B Sweep with Bayesian optimization over 12 PPO hyperparameters.

## User Review Required

> [!IMPORTANT]
> **Optuna vs W&B Sweep**: I'm choosing **Optuna** (local, no account/API key needed, TPE sampler = Bayesian optimization) over W&B. Optuna is already the de-facto standard for RL hyperparameter tuning (used by SB3 Zoo). If you'd prefer W&B instead, let me know and I'll switch.

> [!IMPORTANT]
> **Budget**: At ~10K FPS with 200 envs, each 10M-step trial takes ~17 min. Overnight (~8 hours) → **~24 trials**. I'll default to 10M steps per trial (enough to see learning signal without full convergence, per Joel's strategy of "favoring runs that converge quickly"). The best config can then be re-run at 25M+ for the real deal.

## Search Space

Based on the reference article + SB3 best practices, tuning these 11 PPO hyperparameters:

| Parameter | Search Range | Type | Rationale |
|-----------|-------------|------|-----------|
| `learning_rate` | {3e-5, 1e-4, 3e-4, 1e-3} | categorical | Most important per reference — controls stability |
| `n_steps` | {500, 750, 1000, 1500} | categorical | Steps per env per update (×200 envs = rollout buffer size) |
| `n_epochs` | {5, 10, 15, 20} | categorical | PPO epochs per update — too many → overfitting on rollout |
| `gamma` | U(0.95, 0.999) | float | Discount — affects how far the policy plans ahead |
| `gae_lambda` | U(0.9, 1.0) | float | GAE bias-variance tradeoff |
| `clip_range` | {0.1, 0.2, 0.3} | categorical | PPO clipping — smaller = more conservative updates |
| `ent_coef` | {0.0, 0.001, 0.005, 0.01, 0.02, 0.05} | categorical | Exploration vs exploitation |
| `vf_coef` | {0.5, 1.0} | categorical | Value function loss weight |
| `max_grad_norm` | {0.3, 0.5, 1.0} | categorical | Gradient clipping |
| `net_arch` | {[128,128], [256,256], [256,128], [128,128,128]} | categorical | Network architecture |
| `activation_fn` | {Tanh, ReLU} | categorical | Activation function |

**Not tuned** (fixed): reward YAML config (locked to current baseline), device=cpu, VecNormalize settings, seed (fixed at 42 for sweep — multi-seed comes later for validation).

**batch_size** is derived from `n_steps × num_envs` (full-batch updates) or a fraction thereof. I'll compute it automatically.

## Proposed Changes

### Hyperparameter Sweep Module

New directory: `flightrl/training/hp_sweep/`

---

#### [NEW] [hp_sweep_optuna.py](file:///home/ssen/flightmare/flightrl/training/hp_sweep/hp_sweep_optuna.py)

Main Optuna orchestrator:
- Creates an Optuna study with TPE sampler + MedianPruner
- For each trial: samples hyperparameters → writes them to a JSON file → spawns `_train_one_hp.py` as subprocess (same pattern as existing sweep)
- Subprocess writes `result.json` with final metrics
- Objective: maximize `eval_mean_reward` (or `eval_max_gate_idx` as primary, reward as tiebreaker — composite score)
- Saves Optuna study database as SQLite for resume capability
- Prints running best + leaderboard after each trial
- **Time-based termination**: auto-stops after `--max-hours` (default 8h) to avoid running past morning

#### [NEW] [_train_one_hp.py](file:///home/ssen/flightmare/flightrl/training/hp_sweep/_train_one_hp.py)

Per-trial training subprocess (adapted from existing `sweep/_train_one.py`):
- Reads hyperparameters from a JSON file passed as argument
- Creates PPO with sampled hyperparameters
- Trains for `--total-timesteps` steps
- Runs deterministic eval from start_pos (10 episodes)
- Writes `result.json` with all metrics + hyperparameters
- Writes `progress.json` for learning curve analysis

#### [NEW] [launch_hp_sweep.sh](file:///home/ssen/flightmare/flightrl/training/hp_sweep/launch_hp_sweep.sh)

nohup launcher script (same pattern as existing `sweep/launch.sh`):
- Activates conda, exports paths
- Runs orchestrator under nohup with logging
- Prints PID + tail command for monitoring

#### [NEW] [hp_sweep_report.py](file:///home/ssen/flightmare/flightrl/training/hp_sweep/hp_sweep_report.py)

Morning analysis script:
- Reads Optuna study DB + per-trial result.json files
- Prints ranked table of all trials
- Plots hyperparameter importance (Optuna's built-in fANOVA)
- Generates comparison plot of learning curves
- Saves best hyperparameters to `best_hparams.json`

## Verification Plan

### Automated Tests
1. **Smoke test**: Run sweep with `--total-timesteps 50000 --max-trials 2 --max-hours 0.1` to verify full pipeline works end-to-end
2. Confirm `result.json` files are written with correct schema
3. Confirm Optuna DB is created and can be read back

### Manual Verification
- Launch with `bash hp_sweep/launch_hp_sweep.sh` before bed
- Morning: run `python hp_sweep/hp_sweep_report.py` to see results
- Re-train best config at full 25M steps to validate
