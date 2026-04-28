# Contributing to Flightmare Drone Racing

Thanks for wanting to contribute! This guide will help you get set up and start working on the project.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/CornellPhysicalIntelligence/flightmare.git
cd flightmare
export FLIGHTMARE_PATH=$(pwd)
echo "export FLIGHTMARE_PATH=$(pwd)" >> ~/.bashrc
```

### 2. Create Conda Environment

```bash
conda create -n flightmare python=3.8 -y
conda activate flightmare
pip install stable-baselines3[extra] torch numpy matplotlib ruamel.yaml pyyaml
```

### 3. Build the C++ Physics Engine

```bash
cd flightlib
python setup.py build_ext --inplace
```

> **Important:** You must rebuild after any changes to `.cpp` or `.hpp` files:
> ```bash
> cd ~/flightmare/flightlib
> conda activate flightmare && export FLIGHTMARE_PATH=~/flightmare
> python3 setup.py build_ext --inplace
> ```

### 4. Verify Installation

```bash
cd flightrl/training
python diag.py  # Quick sanity check (obs dims, reward range, FPS)
```

---

## Project Architecture

```
flightlib/          C++ physics engine (compiled to flightgym.so via pybind11)
  ├── src/envs/     Environment implementations (quadrotor_env.cpp)
  ├── include/      Headers
  ├── configs/      YAML configs (track layout, env params)
  └── tests/        C++ tests

flightrl/training/  Python training & evaluation
  ├── gate_racing_iros.py       Main training script (start here)
  ├── sb3_vec_env.py            SB3 VecEnv wrapper
  ├── visualize_policy.py       Trajectory visualization
  ├── run_ablation.py           Ablation study runner
  └── reward_probe.py           Reward diagnostics
```

---

## What to Work On

### Open Issues (Priorities 6-11)

These are the remaining gaps between our implementation and the IROS 2021 paper. Each is a self-contained task:

| Priority | Task | Difficulty |
|----------|------|------------|
| **P6** | Gate collision terminal penalty | Medium — requires C++ geometry |
| **P7** | Network arch tuning (`[256,256]`, Tanh output) | Easy — Python only |
| **P8** | Gaussian safety reward near gate edges | Medium — C++ reward |
| **P9** | Remove survival bonus, tune gate bonus | Easy — C++ constants |
| **P10** | Gate collision detection + termination | Medium — C++ geometry |
| **P11** | Body rate penalty annealing | Easy — C++ step counter |

Pick one, open an issue to claim it, and submit a PR!

---

## Code Style

### C++
- Follow the existing `.clang-format` configuration
- Run `clang-format -i <file>` before committing

### Python
- Use [Black](https://github.com/psf/black) for formatting: `black --line-length 88 <file>`
- Use type hints where practical
- Docstrings for all public functions

---

## Pull Request Workflow

1. **Fork** the repo and create a feature branch:
   ```bash
   git checkout -b feature/p7-network-arch
   ```

2. **Make your changes** — rebuild C++ if needed

3. **Test locally:**
   ```bash
   # Quick smoke test (should run without errors)
   cd flightrl/training
   python gate_racing_iros.py --train 1 --total-timesteps 10000

   # Run diagnostics
   python diag.py
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "P7: Increase network to [256,256] with Tanh output activation"
   ```

5. **Push** and open a PR against `main`

---

## Training Tips

- **Full training run:** ~25M timesteps, takes ~2-4 hours depending on hardware
- **Quick iteration:** Use `--total-timesteps 500000` for fast feedback
- **Monitor training:** `tensorboard --logdir saved/tb_logs --bind_all`
- **Evaluate:** Always load VecNormalize stats alongside the model

---

## Questions?

Open an issue or reach out in the Cornell Physical Intelligence Slack/Discord.
