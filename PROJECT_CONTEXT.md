# Autonomous Drone Racing — Project Context
**Last updated:** 2026-04-28
**Status:** ✅ Faithful IROS 2021 replication implemented — ready for training & ablation

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
│       ├── quadrotor_env.yaml          # Split-S track (7 gates with quaternion rotations)
│       └── vec_env.yaml                # 100 envs, no render
│
└── flightrl/examples_sb3/             # Python training scripts
    ├── gate_racing_iros.py             # Main training + evaluation script (VecNormalize)
    ├── sb3_vec_env.py                  # SB3 VecEnv wrapper (with getObs fix)
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
- **Terminal conditions:** z ≤ 0.02m (ground crash) or |pos| > 100m (OOB)

### Track: Split-S (7 gates)
From the paper's original track files. Features vertically stacked gates (gates 4 & 5) requiring rapid elevation changes. Gate rotations stored as quaternions `[w, x, y, z]` in YAML.

### Reward Function (Paper Eq. 2-5)
```
r(t) = progress + gate_bonus + survival + control_penalty + body_rate_penalty
```
- **Progress:** `min(s_curr − s_prev, δ_max)` — projected distance along segment
- **Gate bonus:** `+10.0` on gate pass
- **Survival:** `+0.1` per step
- **Control penalty:** `-0.0001 · ‖a‖²`
- **Body rate penalty:** `-0.0005 · ‖ω‖²`
- **Crash penalty:** `-5.0` (flat, on ground crash or OOB)

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
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    n_steps=250,         # per env → 25K total per update
    batch_size=25000,    # full-batch updates
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # entropy bonus for exploration
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
cd ~/flightmare/flightrl/examples_sb3
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

### Bug 1: `-Ofast` breaks `std::isfinite()` — physics never advances
`-Ofast` enables `-ffinite-math-only` which makes `std::isfinite(NaN)` return `true`, corrupting `Command::valid()`. The drone never moved. **Fix:** volatile-based NaN checks in `command.cpp`.

### Bug 2: Pybind11 `Ref<MatrixRowMajor>` row expression creates a temporary
`obs.row(i)` creates a temp copy — writes inside `step()` discarded. **Fix:** exposed `getObs()` in pybind, called explicitly after `step()` in `sb3_vec_env.py`.

### Bug 3: `EvalCallback` hangs forever
VecEnv auto-resets before Python sees `done=True`. **Fix:** replaced with custom `TrainingRewardCallback`.

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
