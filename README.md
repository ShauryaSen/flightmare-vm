# Flightmare — Autonomous Drone Racing (IROS 2021 Replication)

**Cornell Physical Intelligence** · Drone Racing RL Research

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Replicating and extending [Autonomous Drone Racing with Deep Reinforcement Learning (IROS 2021)](https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf) using [Flightmare](https://github.com/uzh-rpg/flightmare) + Stable-Baselines3 PPO.

---

## Overview

This is a fork of [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) focused on:

1. **Faithful replication** of the IROS 2021 autonomous drone racing baseline (Song et al.)
---

## Project Structure

```
flightmare/
├── flightlib/                              # C++ physics engine → compiled into flightgym.so
│   ├── src/envs/quadrotor_env/
│   │   └── quadrotor_env.cpp               # IROS 2021 env (26-dim obs, spherical gates)
│   ├── include/flightlib/envs/quadrotor_env/
│   │   └── quadrotor_env.hpp               # Header with obs space enums + TrajectorySnapshot
│   ├── src/common/
│   │   └── command.cpp                     # -Ofast NaN fix (volatile-based)
│   └── configs/
│       ├── quadrotor_env.yaml              # Split-S track (7 gates, quaternion rotations)
│       └── vec_env.yaml                    # 100 envs, no render
│
├── flightrl/training/                      # Python training + evaluation scripts
│   ├── gate_racing_iros.py                 # Main training script (PPO + VecNormalize)
│   ├── sb3_vec_env.py                      # SB3 VecEnv wrapper for flightgym
│   ├── run_ablation.py                     # Ablation study runner
│   ├── sb3_vec_env_ablation.py             # Ablation-aware VecEnv wrapper
│   ├── visualize_policy.py                 # Trajectory recording + plotting + GIF
│   ├── reward_probe.py                     # Reward signal diagnostic tool
│   └── verify_reward.py                    # Reward calculation verification
│
├── IROS21_tex_source/                      # Original paper LaTeX source (reference)
├── CONTRIBUTING.md                         # How to contribute
└── README.md                               # This file
```

---

## Quick Start

### Prerequisites

- Ubuntu 20.04+ (tested on 24.04 ARM64)
- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda
- CMake 3.16+, GCC/G++ with C++17 support

### 1. Clone & Setup Conda Environment

```bash
git clone https://github.com/CornellPhysicalIntelligence/flightmare.git
cd flightmare
export FLIGHTMARE_PATH=$(pwd)

conda create -n flightmare python=3.8 -y
conda activate flightmare
pip install stable-baselines3[extra] torch numpy matplotlib ruamel.yaml pyyaml
```

### 2. Build the C++ Physics Engine

```bash
cd flightlib
pip install .
# or for development:
python setup.py build_ext --inplace
```

### 3. Train

```bash
cd flightrl/training
python gate_racing_iros.py --train 1 --total-timesteps 25000000
```

### 4. Evaluate

```bash
python gate_racing_iros.py --train 0 \
  --model-path saved/best_iros_gate_racing/best_model.zip \
  --vecnorm-path saved/best_iros_gate_racing/vec_normalize.pkl
```

### 5. Visualize

```bash
python visualize_policy.py \
  --model saved/best_iros_gate_racing/best_model.zip \
  -n 30 --animate --animate-n 3
```

---

## Environment Specification (IROS 2021)

### Observation Space (26-dim)

| Component | Dims | Description |
|-----------|------|-------------|
| Linear velocity | 3 | `v_WB` in world frame |
| Linear acceleration | 3 | `v̇_WB` via finite difference |
| Rotation matrix | 9 | Flattened 3×3 `R_WB` (no gimbal lock) |
| Angular velocity | 3 | `ω_B` in body frame |
| Gate 1 (next) | 4 | Spherical `(r, θ, φ, α)` in body frame |
| Gate 2 (future) | 4 | Spherical `(r, θ, φ, α)` in gate 1's frame |

**Key design:** No absolute position (position-invariant policy), rotation matrix instead of Euler angles, spherical gate coordinates.

### Action Space (4-dim)

Per-motor normalized thrusts `[T1, T2, T3, T4]`.

### Reward Function

```
r(t) = progress + gate_bonus + survival - control_penalty - body_rate_penalty
```

---

## Current Status

### ✅ Implemented (Paper Priorities 1-5)

- [x] Full 26-dim observation space (vel, accel, rotmat, ω, 2 spherical gates)
- [x] Spherical gate representation with gate normal angle
- [x] 2 future gates in observation
- [x] VecNormalize z-score observation normalization
- [x] Distributed initialization (hover → trajectory replay)

### 🔲 TODO (Priorities 6-11)

- [ ] **P6:** Distance-dependent terminal penalty for gate collisions
- [ ] **P7:** Network architecture tuning (Tanh output, `[256,256]`)
- [ ] **P8:** Gaussian safety reward near gate edges
- [ ] **P9:** Remove survival bonus (conflicts with time-optimal objective)
- [ ] **P10:** Gate collision detection + termination
- [ ] **P11:** Body rate penalty annealing schedule

See [`iros21_replication_gaps.md`](iros21_replication_gaps.md) for detailed descriptions of each item.

---

## Critical Bugs Found & Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `-Ofast` breaks `std::isfinite(NaN)` | Physics never advances | Volatile-based NaN checks in `command.cpp` |
| Pybind11 `Ref<MatrixRowMajor>` row expression creates temporary | Obs writes discarded | Exposed `getObs()` in pybind, called after `step()` |
| `EvalCallback` hangs on C++ VecEnv | Training stalls | Custom `TrainingRewardCallback` |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and contribution guidelines.

---

## Citations

If you use this code, please cite:

```bibtex
@inproceedings{song2021autonomous,
    title={Autonomous Drone Racing with Deep Reinforcement Learning},
    author={Song, Yunlong and Scaramuzza, Davide},
    booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2021}
}

@inproceedings{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```

## License

This project is released under the [MIT License](LICENSE).
