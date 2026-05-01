# Autonomous Drone Racing — Project Context
**Last updated:** 2026-05-01
**Status:** ✅ Spline-based reward system validated — training with deviation-penalized progress reward

---

## 1. Project Goal

Faithfully replicate the **IROS 2021 autonomous drone racing baseline** (Song et al.) in [Flightmare](https://github.com/uzh-rpg/flightmare) using Stable-Baselines3 PPO. Once the baseline matches the paper's performance, conduct an **observation-space ablation study** to characterize which state components are necessary, redundant, or load-bearing for agile gate racing.

**Paper:** [Autonomous Drone Racing with Deep Reinforcement Learning (IROS 2021)](https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf)

### Research Question

> **"What is the minimal sufficient observation space for an RL drone racing policy to reliably traverse a sequence of gates, and how does each observation component contribute to learning efficiency, peak performance, and behavioral stability?"**

### Target Venues
- **Near-term:** CoRL 2025 / IROS 2025 workshop (class results + 3-4 ablations)
- **Full paper:** IROS 2026 (full 5-seed study over summer)
- **Backup:** RA-L (rolling, good for characterization studies)

---

## 2. Infrastructure

### VM (all training / simulation / development)
| Property | Value |
|----------|-------|
| OS | Ubuntu 24.04 ARM64 (Apple Silicon VM via UTM) |
| Conda env | `flightmare` (Python 3.8) |
| Flightmare root | `~/flightmare` |
| Training scripts | `~/flightmare/flightrl/training/` |
| `FLIGHTMARE_PATH` | `~/flightmare` (in `~/.bashrc`) |

> [!IMPORTANT]
> **Development is done directly on the VM via Remote-SSH in VS Code.**

### Key Paths on VM
```
~/flightmare/
├── flightlib/                          # C++ physics engine → compiled into flightgym.so
│   ├── src/envs/quadrotor_env/
│   │   └── quadrotor_env.cpp           # IROS 2021 env (26-dim obs, spherical gates, distributed init)
│   ├── include/flightlib/envs/quadrotor_env/
│   │   └── quadrotor_env.hpp           # Header with obs space enums + TrajectorySnapshot struct
│   ├── src/common/
│   │   └── command.cpp                 # -Ofast NaN fix (volatile-based)
│   ├── src/wrapper/
│   │   └── pybind_wrapper.cpp          # getObs() + getEpisodeLength() exposed
│   └── configs/
│       ├── quadrotor_env.yaml          # AlphaPilot track (11 gates with quaternion rotations)
│       └── vec_env.yaml                # 100 envs, no render
│
└── flightrl/training/                 # Python training scripts
    ├── gate_racing_iros.py             # Main training + evaluation script (VecNormalize)
    ├── sb3_vec_env.py                  # SB3 VecEnv wrapper (with getObs fix)
    ├── reward_math.js                  # JS port of C++ reward math (for visualizer)
    ├── reward_debugger.html            # 3D interactive reward visualizer (Three.js)
    ├── run_ablation.py                 # Ablation study runner
    ├── sb3_vec_env_ablation.py         # Ablation-aware VecEnv wrapper
    ├── visualize_policy.py             # Trajectory recording + plotting + GIF animation
    ├── diag.py                         # Quick env diagnostic script
    └── saved/
        ├── best_iros_gate_racing/      # Best model + VecNormalize stats
        ├── checkpoints/               # Every 500K steps
        └── tb_logs/                   # TensorBoard logs
```

### Rebuild C++ (after any .cpp/.hpp changes)
```bash
cd ~/flightmare/flightlib
source ~/miniforge3/etc/profile.d/conda.sh && conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare
python3 setup.py build_ext --inplace
```

---

## 3. Environment Specification (IROS 2021 Faithful)

### Observation Space (26-dim)

The observation matches the paper (methodology.tex L173, L183-191):

```
s_quad  = [v_WB(3), v̇_WB(3), R_WB(9), ω_B(3)]      = 18 dims (proprioception)
s_track = [o1(r,θ,φ)(3), α1, o2(r,θ,φ)(3), α2]       =  8 dims (2 gates × 4 each)
Total   = 26 dims
```

| Component | Dim | Index | Description |
|-----------|-----|-------|-------------|
| Linear velocity | 3 | `[0:3]` | `v_WB` in world frame |
| Linear acceleration | 3 | `[3:6]` | `v̇_WB` via finite difference `(v_curr − v_prev) / dt` |
| Rotation matrix | 9 | `[6:15]` | Flattened 3×3 `R_WB` (row-major) — avoids Euler gimbal lock |
| Angular velocity | 3 | `[15:18]` | `ω_B` in body frame |
| Gate 1 (next) | 4 | `[18:22]` | Spherical `(r, θ, φ)` in body frame + gate normal angle `α` |
| Gate 2 (2nd future) | 4 | `[22:26]` | Spherical `(r, θ, φ)` in gate 1's frame + gate normal angle `α` |

**Key design choices (from the paper):**
- **No absolute position** — policy is position-invariant, generalizes across track segments
- **Rotation matrix instead of Euler angles** — avoids gimbal lock during aggressive maneuvers
- **Linear acceleration** — provides force/dynamics information critical for anticipatory control
- **Spherical gate coords** — separates distance from direction (paper L187)
- **2 future gates** — enables approach angle planning (paper experiments.tex L48)
- **Gate normal angle α** — tells policy which direction to fly through each gate

### Action Space (4-dim)
Per-motor thrusts, normalized: `act = action * act_std + act_mean`

### Physics
- **Sim dt:** 0.02s (50 Hz)
- **Integration:** RK4 rigid-body dynamics, quadratic thrust model
- **Terminal conditions:** z ≤ 0.02m (ground crash), |pos| > 100m (OOB), or t ≥ 30s (timeout)
- **No gate collision termination** — only progress-based reward; gate pass detection advances the target gate index but does not terminate

### Track: AlphaPilot (11 gates)
From the paper's original track files with spline-derived gate orientations. Gate rotations stored as quaternions `[w, x, y, z]` in YAML. A cubic Hermite spline connects `[start_pos, G0, G1, ..., G10]` with tangent vectors matching gate fly-through normals. Total arc length: ~213m. Start position: `[17.4, -5.0, 6.31]`, gate width: 5m.

### Reward Function (Spline-based, evolved from Paper Eq. 4-5)
```
r(t) = r_progress(t) + r_deviation(t) + a·r_safety(t) − b·‖ω‖²
```

#### 1. Spline Progress (`r_progress`)
The core reward signal. A cubic Hermite spline is built through `[start_pos, gate0, ..., gateN]` with tangent vectors matching gate fly-through normals (from quaternion rotations), scaled by chord length for smooth C1 interpolation. Each segment is densely sampled (200 points) with cumulative arc-length.

A progress score function is defined at each position:
```
s(pos) = (arc_to_nearest_spline_point − segment_start) − deviation_ratio × lateral_distance
```
The per-step reward is the score delta:
```
r_progress = s(p_t) − s(p_{t-1})
```
- **Forward movement along spline → positive reward** (arc increases)
- **Backward movement → negative reward** (arc decreases)
- **Drifting laterally → one-time cost** when deviation changes (ratio = 1.5)
- Score is clamped to `[seg.s_start, seg.s_end]` per segment

**Key files:** `spline_progress()` in [quadrotor_env.cpp](flightlib/src/envs/quadrotor_env/quadrotor_env.cpp), `buildTrackSpline()` for spline construction.

#### 2. Per-Step Deviation Penalty (`r_deviation`)
The progress score's linear deviation term only penalizes *changes* in deviation — a drone flying parallel to the spline at constant 12m offset gets zero deviation cost per step. This quadratic penalty fixes that:
```
r_deviation = −coeff × max(0, dev − gate_width/2)²
```
- **Threshold:** `gate_width/2 = 2.5m` — no penalty within gate radius
- **Quadratic scaling:** soft near the track, harsh far away (at 10m excess: −1.0/step)
- **coeff:** 0.01 (configurable via `deviation_penalty_coeff` in YAML)
- **Fires every step** — cannot be amortized away like the linear term

#### 3. Safety (`r_safety`) — currently disabled
- Paper Eq. 3, gated by `safety_coeff > 0` (default 0)

#### 4. Body Rate Penalty
- `−0.0005 · ‖ω‖²` annealed linearly to 0 over 5M steps

#### What's NOT in the reward:
- **No gate collision termination** — removed; was causing false positives on curved tracks
- **No gate-pass bonus** — gate detection only advances target gate index
- **No** survival bonus, action penalty, or flat crash penalty

### Distributed Initialization (Paper L247-261)
Two-phase initialization strategy — one of the paper's "three key ingredients":
- **Phase 1:** Drones spawn at hover across ALL track segments uniformly (random segment, random interpolation along it)
- **Phase 2:** Once gate-pass rate > threshold (0.3) and trajectory buffer > 100 snapshots, sample initial states from recorded trajectories with realistic velocities
- Snapshots recorded every 10 steps during episodes (if drone is flying, not crashed)

### Observation Normalization
- **VecNormalize** wrapper (SB3) applies running z-score normalization to observations (paper L198-199)
- Stats saved alongside model for proper evaluation

---

## 4. Training Configuration

```python
PPO(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=Tanh),
    n_steps=1500,        # per env → 150K total per update
    batch_size=150000,   # full-batch updates
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
```

100 parallel environments. VecNormalize for obs z-scoring.

**Callbacks:**
- `TrainingRewardCallback` — tracks `ep_rew_mean`, saves best model + VecNormalize stats
- `CheckpointCallback` — saves every 500K steps

### Running Training
```bash
cd ~/flightmare/flightrl/training
conda activate flightmare && export FLIGHTMARE_PATH=~/flightmare

# Full 25M-step run
python -u gate_racing_iros.py --train 1 --total-timesteps 25000000

# With nohup (recommended)
nohup python -u gate_racing_iros.py --train 1 --total-timesteps 25000000 \
  > logs/iros_full.log 2>&1 &
tail -f logs/iros_full.log
```

### Evaluation
```bash
python gate_racing_iros.py --train 0 \
  --model-path saved/best_iros_gate_racing/best_model.zip \
  --vecnorm-path saved/best_iros_gate_racing/vec_normalize.pkl
```

### Visualization
```bash
python visualize_policy.py \
  --model saved/best_iros_gate_racing/best_model.zip \
  -n 30 --animate --animate-n 3
```

### Diagnostics
```bash
python diag.py  # quick env sanity check (obs dims, reward range, FPS)
```

### TensorBoard
```bash
tensorboard --logdir saved/tb_logs --bind_all --port 6006
```

---

## 5. Critical Bugs Found & Fixed

See [FIXES_ATTEMPTED.md](FIXES_ATTEMPTED.md) for the full append-only log with detailed hypotheses, file changes, and outcomes.

### Bug 1: `-Ofast` breaks `std::isfinite()` — physics never advances
`-Ofast` enables `-ffinite-math-only` which makes `std::isfinite(NaN)` return `true`, corrupting `Command::valid()`. The drone never moved. **Fix:** volatile-based NaN checks in `command.cpp`.

### Bug 2: Pybind11 `Ref<MatrixRowMajor>` row expression creates a temporary
`obs.row(i)` creates a temp copy — writes inside `step()` discarded. **Fix:** exposed `getObs()` in pybind, called explicitly after `step()` in `sb3_vec_env.py`.

### Bug 3: `EvalCallback` hangs forever
VecEnv auto-resets before Python sees `done=True`. **Fix:** replaced with custom `TrainingRewardCallback`.

### Bug 4: Progress reward double-clamped (2026-04-30)
Both `segment_progress()` and `step()` clamped progress, zeroing the gradient on longer segments. **Fix:** unclamped raw projection per paper Eq. 2.

### Bug 5: Non-paper reward shaping (2026-04-30)
Removed `gate_bonus`, `survival_bonus`, `act_coeff`, `crash_penalty` — paper has none of these.

### Bug 6: Inertia ~2.5× heavier than paper (2026-04-30)
Auto-computed `J` was too heavy for agile flight. **Fix:** explicit `inertia: [0.003, 0.003, 0.005]` override.

### Bug 7: Linear progress on curved tracks (2026-05-01)
Scalar projection onto straight gate-to-gate lines rewarded backward flight and perpendicular motion on curved tracks. **Fix:** replaced with cubic Hermite spline-based progress (`buildTrackSpline()` + `spline_progress()`). ✅ Validated — monotonic reward improvement during training.

### Bug 8: Gate collision false positives (2026-05-01)
Gate-plane crossing on an infinite plane triggered spurious collisions 50m from gates. Three sub-bugs fixed (proximity guard, rectangular check, interpolated crossing point), then **gate collision termination removed entirely** — it's not needed for progress-based learning and was actively harmful on curved tracks. Terminal conditions: ground crash, OOB, timeout only.

### Bug 9: Constant-offset deviation loophole (2026-05-01)
The linear deviation term inside `spline_progress()` only penalizes *changes* in deviation — maintaining a constant large offset costs nothing per step. Drone could accumulate +42 reward while 12m off-track. **Fix:** (a) increased `spline_deviation_ratio` 0.5→1.5, (b) added per-step quadratic deviation penalty `−0.01 × max(0, dev − gate_width/2)²` — fires every step the drone is beyond gate radius.

---

## 6. Ablation Study Design

### Experiments (each condition: 25M steps, 5 seeds, MLP [256,256])

| ID | Name | Obs Dims | What Changes |
|----|------|----------|-------------|
| A | Full Baseline (control) | 26 | All components |
| B | No Orientation | 17 | Remove R_WB(9) |
| C | No Angular Velocity | 23 | Remove ω_B(3) |
| D | No Gate Observation | 18 | Remove gate obs(8) |
| E | Gate + Velocity Only | 6 | Only lin_vel(3) + gate1(3) |
| F | No Position | 26 | Already done — no pos in obs |
| G1-G4 | Noisy Gate | 26 | Gate obs + N(0, σ²) at σ=0.1,0.3,0.5,1.0 |
| H | Replace Euler w/ Rotmat | 26 | Already done — rotmat used |
| I | IMU-Realistic | 9 | Only lin_vel(3) + ang_vel(3) + gate1(3) |

**Total compute:** ~12 conditions × 5 seeds × 12 min ≈ 12 hours VM time.

### Key Claims to Test
1. Angular velocity is the load-bearing term (Exp C → largest degradation)
2. Gate observation is necessary but noise-tolerant up to σ≈0.3m (Exp D, G)
3. Rotation matrix is partially redundant (Exp B → moderate degradation)
4. Minimal deployable obs = `[lin_vel | ang_vel | gate]` (Exp I)

---

## 7. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| SB3 PPO instead of original TF1 | TF1 incompatible with Python 3.8+ and Ubuntu 24.04 |
| 100 parallel envs | Maximizes throughput (~28K FPS on ARM64) |
| `TrainingRewardCallback` not `EvalCallback` | EvalCallback hangs due to VecEnv auto-reset (Bug 3) |
| Rotation matrix not Euler angles | Paper explicitly avoids gimbal lock (methodology.tex L175) |
| Spherical gate coords not Cartesian | Separates distance from direction (paper L187) |
| 2 future gates not 1 | Paper default; Table III shows 23% crash with 1 gate vs 2.5% with 2 |
| Distributed init across segments | Paper's "key ingredient"; avoids zero training signal on far segments |
| VecNormalize z-scoring | Paper L198-199; prevents gradient domination by large-scale features |
| Per-motor thrusts (not body rates) | Simpler action space; matches paper's setup |
| `getObs()` after `step()` | Workaround for pybind11 Ref temporary (Bug 2) |
| Volatile NaN check in command.cpp | `-Ofast` breaks `std::isfinite(NaN)` (Bug 1) |

---

## 8. Reference Papers
- [IROS 2021 Drone Racing](https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf) — baseline being replicated
- [Flightmare (RA-L 2021)](https://arxiv.org/abs/2009.00563) — simulator
- [Swift (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4) — champion-level racing, uses CTBR action space
- [Benchmark Comparison of Learned Controllers (ICRA 2022)](https://rpg.ifi.uzh.ch/docs/ICRA22_Elia.pdf) — action space comparison
- [On the Continuity of Rotation Representations (CVPR 2019)](https://arxiv.org/abs/1812.07035) — rotation matrix > Euler for NNs
