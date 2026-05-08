# Experiment Status — Where We Are vs. EXPERIMENT_PLAN.md

**Last updated:** 2026-05-08
**Companion to:** [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) (the spec) — this file tracks
execution progress against it.

**Headline:** All scaffolding done. Cloud VM provisioned. Ready to bootstrap and launch.
The only thing standing between us and the 9-run campaign is running `cloud_setup.sh` on
the GCP box and verifying the four smoke tests on x86.

---

## TL;DR — what's done vs. what's left

```
[████████████████████░░░░] ~85% complete

DONE     ✓ Local-prep + C++ pybind + Condition C rewrite + smoke tests + pilot
DONE     ✓ Cloud VM provisioned (Debian 12 bookworm x86_64)
NEXT     ↳ SSH in → bash cloud_setup.sh → smoke tests on cloud
THEN     ↳ tmux + bash run_all_experiments.sh  (~9–12 hr unattended)
WRAP     ↳ rsync results back → stop the instance → write the paper
```

---

## What's Done

### Phase 1 — Local prep ✓ (commit `75584de`)
- `experiment_hparams.json` locked to §4 sweep-winner HPs (`batch_size: 100000` to keep full-batch updates with 200 envs).
- `_run_lib.sh`, `run_condition_A.sh`, `run_condition_B.sh` (6-rung warm-start handoff).
- `pilot_C_5M.sh`, `run_all_experiments.sh` orchestrator.

### Phase 2 — C++ pybind hook ✓
- `applyPerturbedGates(env_id, positions, yaws_rad)` — mutates per-env gates in-process,
  rebuilds spline, resets `s_prev_`.
- `getGates(env_id, positions)` + `getNumGates(env_id)` — round-trip introspection for
  smoke tests.
- Implemented in [quadrotor_env.cpp](flightlib/src/envs/quadrotor_env/quadrotor_env.cpp),
  exposed via [pybind_wrapper.cpp](flightlib/src/wrapper/pybind_wrapper.cpp). Built and
  verified with `smoke_pybind.py` round-trip test.

### Phase 3 — Condition C rewritten to match IROS21 §III.D ✓ (commit `855a02e`)

The σ-perturbed-circle_8 v1 design was wrong (started the curriculum at the target's own
complexity). Replaced with the paper's actual approach:
- **`linear_8.yaml`** — straight-line baseline (`start_pos=(0,0,5)`, 8 gates marching
  forward at 6 m intervals along +x, all yaw=0). The `c=0` rung.
- **`track_generator.py`** — concatenates gate primitives `G_{i+1} = f(G_i, Δp, ΔR)` in
  the previous gate's local frame; complexity `c ∈ [0, 1]` linearly scales per-step
  Δyaw (35°), lateral (4 m), vertical (1.5 m).
- **`adaptive_complexity.py`** — additive `c ± 0.05` step every 100 episodes based on
  completion ratio (>0.7 → up; <0.3 → down). Floor c=0 (so we don't multiplicatively
  trap there); ceiling c=1.
- **`random_track_train.py`** — extends `_train_one_hp.py` with a `RandomTrackCallback`
  that resamples each env's track on `done[i]==True` and pushes the completion signal
  into the controller.

**See:** [track_previews_random/_overview_grid.png](flightrl/training/experiment/track_previews_random/_overview_grid.png)
for the c-progression visualization.

### Phase 4 — Eval grid ✓
- `_eval_one.py` runs a single (model × track [× perturb]) eval as a subprocess.
- Now supports **`--procedural-c`** + **`--procedural-seed`**: applies `linear_8`
  baseline, then overlays a TrackGenerator draw before deterministic rollouts. Same
  draw replays across all 9 condition checkpoints (so per-episode bootstrap CIs are
  valid).
- `run_eval_grid.py` orchestrates two passes:
  1. Named-track held-out (`circle_8` target + `oval_8/helix_8/vertical_3/figure_8`
     held-out + each perturbed at σ=0.5) — closed-loop tracks, fair for A/B.
  2. Procedural held-out — 5 fixed seeds × `c ∈ {0.7, 0.85, 1.0}` = 15 pairs per
     checkpoint — open tracks, fair for C.
- 95% bootstrap CIs (10 000 resamples) computed per (condition, eval pair) pooled
  across seeds.

### Phase 5 — Cloud bootstrap ✓
- `cloud_setup.sh` at repo root: apt deps → miniforge → conda env from
  `environment.yml` → build flightlib → run `diag.py` → write `RUN_MANIFEST.md`
  (git SHA + `flightgym.so` SHA-256 + frozen env).
- `environment.yml` — cross-arch (auto-export from local was ARM-pinned and would
  have failed on x86; hand-curated to minimal conda + pinned pip stack).
- `vec_env.yaml: render: no` — disables the Unity ZMQ bridge so multiple flightgym
  processes coexist (smoke + viz + eval grid). Required for the eval grid's
  sequential-subprocess pattern.

### Phase 6 — Smoke verification on local UTM ✓
All four pass on the ARM64 box (commit `855a02e`):
- `smoke_pybind.py` — round-trip + per-env independence + VecNormalize attribute path
- `smoke_A.sh 0` — Condition A wiring (100K steps; ~9 s)
- `smoke_B.sh 0` — Condition B 6-rung handoff (~54 s total; warm-start verified)
- `smoke_C.sh 0` — Condition C generator + controller (200K steps; ~18 s)

### Pilot validation — Condition C
Two local pilots confirm the curriculum mechanic works as designed and forecast cloud
behavior:

| Pilot   | Steps | Wall  | Peak c | Final c | Mean gates | c-bumps |
|---------|------:|------:|-------:|--------:|-----------:|--------:|
| 5M      |  5e6  | 7 min |   0.15 |    0.15 |     5.80/8 |     67  |
| 20M     |  2e7  | 32 min|   0.30 |    0.20 |     5.99/8 |     68  |

End-policy rollouts on the 20M model:
- **c=0.0**: 8/8, 8/8, 8/8 (perfect on the literal straight-line baseline)
- **c=0.2**: 8/8, 8/8, 8/8 (training distribution — perfect)
- **c=0.5**: 1/8, 3/8, 3/8 (out-of-distribution — degrades)
- **c=1.0**: 1/8, 5/8, 1/8 (mostly fails)

The policy is dialed in for the complexity it *actually* trained on but doesn't
generalize beyond it. Honest IROS21-style finding: the curriculum tracks the agent's
competence ceiling rather than guaranteeing it reaches `c=1.0`.

**Implication for the cloud campaign:** projecting linearly, 100M won't push `c` above
~0.4. The plan §3 wording is already tempered to "complexity reached during training"
rather than "full complexity."

---

## What's Left

### Cloud bootstrap (sequential, ~1 hr including the build)

The user has already provisioned the GCP `c3-standard-8` VM (Debian 12 bookworm
x86_64). Remaining:

1. **SSH into the VM and clone the repo.**
   ```bash
   gcloud compute ssh <instance> --zone=<zone>
   sudo apt-get update -y && sudo apt-get install -y git
   git clone git@github.com:ShauryaSen/flightmare-vm.git ~/flightmare
   cd ~/flightmare && git log --oneline -1
   # Should show: 855a02e Lock cloud-launch state: render off, env.yml, plan v4, procedural eval
   ```
   If the private repo blocks SSH-key auth on the box, easiest is a GitHub deploy
   key (read-only is fine).

2. **Run `cloud_setup.sh`.**
   ```bash
   cd ~/flightmare
   bash cloud_setup.sh 2>&1 | tee cloud_setup.log
   ```
   Installs apt deps → miniforge → conda env → builds flightlib → runs `diag.py` →
   writes `RUN_MANIFEST.md`. ~10–20 min on first run. **If `diag.py` fails, stop.**
   That's the bootstrap's own smoke test.

3. **Lower OMP threads to match the 8 vCPUs.** The repo ships
   `vec_env.yaml: num_threads: 14` (tuned for the local UTM where vCPU
   over-allocation works). On `c3-standard-8` that oversubscribes:
   ```bash
   sed -i 's/num_threads: 14/num_threads: 8/' ~/flightmare/flightlib/configs/vec_env.yaml
   ```
   Don't commit — machine-specific.

4. **Run all 4 smoke tests on the cloud.** This is the gate before the campaign —
   local was ARM64, cloud is x86, first-time x86 build can surface issues `diag.py`
   doesn't catch.
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate flightmare
   export FLIGHTMARE_PATH=~/flightmare
   cd ~/flightmare/flightrl/training/experiment

   python smoke/smoke_pybind.py  # ~5 s
   bash smoke/smoke_A.sh 0       # ~10 s
   bash smoke/smoke_B.sh 0       # ~60 s
   bash smoke/smoke_C.sh 0       # ~20 s
   ```
   All four should print `OK`.

5. **Set a billing alert** at https://console.cloud.google.com/billing → Budgets,
   $30 (60% of the $50 class credit). Recommended; not blocking.

### The campaign itself (unattended, ~9–12 hr)

6. **Launch in tmux.**
   ```bash
   tmux new -s train
   cd ~/flightmare/flightrl/training/experiment
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate flightmare
   export FLIGHTMARE_PATH=~/flightmare
   bash run_all_experiments.sh 2>&1 | tee run_all.log
   ```
   `Ctrl-b d` to detach. The job survives SSH disconnects.

   The orchestrator runs `pilot_C_5M.sh` → `A×3 seeds` → `B×3 seeds` → `C×3 seeds` →
   eval grid (named + procedural held-out). It is **idempotent**: any run with
   `final_model.zip` is skipped on re-launch, so you can interrupt and resume.

7. **Reattach to monitor:**
   ```bash
   gcloud compute ssh <instance> --zone=<zone> -- -t tmux attach -t train
   ```
   Or tail logs from local:
   ```bash
   gcloud compute ssh <instance> --zone=<zone> --command='tail -50 ~/flightmare/flightrl/training/experiment/run_all.log'
   ```

### Wrap-up

8. **`rsync` results back, then stop the instance.**
   ```bash
   gcloud compute scp --recurse <instance>:~/flightmare/flightrl/training/experiment/runs ./gcp_runs/ --zone=<zone>
   gcloud compute instances stop <instance> --zone=<zone>
   ```
   Stopped (not deleted) instances bill ~$0.04/day for the disk and re-spin in seconds
   if you need reruns. Delete only after the paper deadline.

9. **Generate paper plots from the local copy of `runs/`** — reward curves with rung
   boundaries marked for B, complexity-trajectory curve for C, generalization bar
   chart with bootstrap CIs (named + procedural held-out, side by side), end-policy
   visualizations.

10. **Write the paper.**

---

## Known Issues to Watch For

These came up during local validation; the cloud-side mitigations are baked into the
scripts but worth knowing:

- **Stale `flightgym.so` shadowing.** Python loads `flightgym` from whichever copy
  appears first on `sys.path`. Locally we had three (`flightlib/`, `flightrl/training/`,
  conda site-packages). On the cloud `cloud_setup.sh` builds in place; if
  `python -c 'import flightgym'` from `flightrl/training/` doesn't see the new
  pybind methods, `cp` the fresh `.so` to all three locations.
- **Unity ZMQ port collision.** Solved by `render: no`; do not re-enable on the cloud
  unless you specifically want Unity (which we don't — the box has no display anyway).
- **Procedural eval requires the `linear_8` carrier.** The eval grid script applies
  it automatically before the procedural pass. If you run `_eval_one.py` directly,
  remember `--track linear_8 --procedural-c <c>`.
- **Condition C compute ceiling.** Empirically validated: 100M is not going to push
  `c` to 1.0. The plan and the paper framing already reflect this.

---

## File Map (for picking back up cold)

```
flightmare/
├── EXPERIMENT_PLAN.md                # ← spec (read this first)
├── STATUS.md                         # ← this file (where we are)
├── cloud_setup.sh                    # one-shot bootstrap on GCP
├── environment.yml                   # cross-arch conda spec
├── flightlib/
│   ├── configs/
│   │   ├── vec_env.yaml              # render: no, num_envs: 200, num_threads: 14 (lower to 8 on cloud)
│   │   └── tracks/linear_8.yaml      # Condition C c=0 baseline
│   ├── include/flightlib/envs/quadrotor_env/quadrotor_env.hpp  # applyPerturbedGates / getGates
│   ├── src/envs/{quadrotor_env/quadrotor_env.cpp, vec_env.cpp} # impls
│   └── src/wrapper/pybind_wrapper.cpp                          # bindings
└── flightrl/training/experiment/
    ├── _run_lib.sh                   # bash helpers
    ├── run_condition_{A,B,C}.sh      # per-condition entry points
    ├── run_all_experiments.sh        # top-level orchestrator
    ├── pilot_C_5M.sh                 # 5M Condition C pre-validation
    ├── experiment_hparams.json       # locked §4 HPs
    ├── linear_8 logic ──┐
    ├── track_generator.py           │ Condition C
    ├── adaptive_complexity.py       │ random-track
    ├── random_track_train.py        │ stack
    ├── visualize_random_tracks.py   │ (track previews)
    ├── visualize_end_policy.py      │ (rollout overlays)
    ├── _eval_one.py                  # single eval subprocess (named OR procedural)
    ├── run_eval_grid.py              # eval grid orchestrator
    └── smoke/                        # 4 smoke tests; all green on local
```

---

## Quick "What do I run?" by situation

| Situation | Command |
|---|---|
| First time on the cloud box | `bash cloud_setup.sh` then the 4 smoke tests |
| Smoke tests passing — ready to launch | `tmux new -s train`, then `bash run_all_experiments.sh` |
| Need to rerender track previews | `python visualize_random_tracks.py` |
| Have a finished checkpoint, want end-policy viz | `python visualize_end_policy.py --model <path> --vecnorm <path> --out-dir <dir>` |
| Want to inspect curriculum trajectory of a finished C run | `cat runs/C_seed*/complexity_history.json \| jq '[.[] \| select(.c_after != .c_before)] \| length'` |
| Pulled results back, want eval grid summary | already in `runs/eval_grid/eval_grid_results.json` after `run_eval_grid.py` finishes |
