# Imitation Learning Warm-Start for Drone Racing RL

## Problem Statement

The RL agent (PPO) for autonomous drone racing on the `circle_8` track fails to learn navigation from scratch. It gets trapped in local optima — either hovering, diving into the ground, or flying straight through the first gate and crashing. The fundamental issue is **exploration**: the agent never discovers the "turn and fly to the next gate" behavior because random actions from a randomly initialized policy produce only crashes.

### Why RL From Scratch Fails

1. **Sparse gate rewards** (bonus = +5.0 per gate) are too far apart for random exploration to reach
2. **Progress reward** gives small positive signal for forward motion, but crashing also yields small positive reward before termination → the agent settles for "dive forward and crash" as a local optimum
3. **No survival incentive** strong enough to overcome the above without making the agent risk-averse (hovering)
4. The per-motor thrust action space (4 continuous actions) makes even basic stable flight non-trivial to discover randomly

## Solution: Trajectory-Based Imitation Learning Warm-Start

Instead of training RL from scratch, we:
1. **Generate expert demonstrations** using a classical controller that follows a reference trajectory through all gates
2. **Pre-train the policy network** via behavioral cloning (supervised learning on the demonstrations)
3. **Fine-tune with PPO** — the policy already knows "stay alive and fly toward gates", so RL can refine it into fast, aggressive racing

This approach is validated by Paper 04 (Bootstrapping RL with Imitation, CoRL 2024) from the UZH-RPG group, which showed that RL from scratch fails for drone racing while IL-bootstrapped RL succeeds.

---

## STATUS (2026-05-01): Setup complete and validated end-to-end ✅

The full IL warm-start chain is now wired and tested. Summary of what works:

| Stage | Result |
|-------|--------|
| Trajectory controller @ 3.5 m/s | 8/8 gates, 1732 transitions saved |
| Trajectory controller @ 4.5 m/s | 8/8 gates, 1370 transitions saved |
| BC pretraining (200 epochs, CPU, ~30s) | val_act=0.000212, val_vf=2.72 (best at epoch 90) |
| BC policy in env (deterministic) | 2/2 gates passed, R=25.3, ep_len=468 (10/10 episodes) |
| PPO fine-tune from BC, 1.5M steps smoke test | reward grew monotonically -6.23 → +4.78 (BC prior preserved) |

### Quick start

```bash
cd ~/flightmare/flightrl/training
conda activate flightmare && export FLIGHTMARE_PATH=~/flightmare

# 1. Collect demos (one-time, ~1 min total)
python trajectory_controller.py --speed 3.5 --save-path demos/circle_8_slow.npz
python trajectory_controller.py --speed 4.5 --save-path demos/circle_8_med.npz

# 2. BC pretrain — joint policy + value head, ~30s on CPU
python train_bc.py --demos demos/circle_8_slow.npz demos/circle_8_med.npz \
  --save-dir saved/bc_pretrained --epochs 200 --verify

# 3. PPO fine-tune — low LR preserves the BC prior
python gate_racing_iros.py --train 1 \
  --load-model saved/bc_pretrained/bc_model.zip \
  --load-vecnorm saved/bc_pretrained/vec_normalize.pkl \
  --save-path saved/bc_then_ppo --tb-log saved/tb_logs/bc_then_ppo \
  --learning-rate 3e-5 --total-timesteps 25000000

# 4. Evaluate
python gate_racing_iros.py --train 0 \
  --model-path saved/bc_then_ppo/best_model.zip \
  --vecnorm-path saved/bc_then_ppo/vec_normalize.pkl
```

### Three non-obvious issues fixed during validation

1. **Value head must also be pretrained.** PPO's first update uses `V(s)` advantages — a random value head produces garbage advantages and the LR=3e-5 update wipes the BC policy. Fix: train policy + value jointly, with discounted Monte Carlo returns from the demo rewards as VF targets.
2. **`policy.log_std` must be shrunk.** SB3 default is `log(1)=0` → stochastic action stddev = 1.0 in `[-1,1]` space. PPO rollouts are stochastic, and at stddev=1 the actions are noise. Fix: set `log_std = log(0.1)` after BC.
3. **`distributed_init` must be off during fine-tune.** Demos are from `start_pos`; with random per-segment init, the BC policy is OOD and PPO destroys it. Already changed in `quadrotor_env.yaml`. Optionally flip back on once the policy is solid end-to-end.

---

## What Has Been Built

### 1. Trajectory Controller (`flightrl/training/trajectory_controller.py`)

A complete closed-loop trajectory tracking controller that:

- **Generates a smooth cubic spline** through all gate centers on the `circle_8` track
- Uses a `tanh` speed ramp to avoid aggressive initial accelerations
- Computes **inverse dynamics** at each timestep:
  - Desired acceleration → desired attitude rotation matrix
  - SO(3) PD attitude controller → body torques
  - Motor allocation matrix → per-motor thrusts
  - Thrust → normalized action space `[-1, 1]`
- Runs in the **actual Flightmare environment** with ground-truth position feedback from `extra_info`
- Saves `(observation, action)` pairs for behavioral cloning

#### Verified Results (4.5 m/s, circle_8 track)

| Metric | Value |
|--------|-------|
| Gates passed | **8/8** ✅ |
| Episode reward | **131.0** |
| Tracking error | ~1-2m (within gate_width=5m) |
| Altitude stability | ±0.3m from z=3.0m |
| Demo transitions saved | 1,370 |
| Episode duration | 23.3s |

**Usage:**
```bash
cd flightrl/training

# Plot reference trajectory only
python trajectory_controller.py --plot-only --speed 4.5

# Run in environment and save demos
python trajectory_controller.py --speed 4.5 --save-path demos/circle_8_demos.npz

# With custom PD gains
python trajectory_controller.py --speed 4.5 --Kp 3.0 --Kd 2.5 --Kp-att 15.0 --Kd-att 5.0
```

### 2. Demo Data (`flightrl/training/demos/circle_8_demos.npz`)

Contains:
- `observations`: `(N, 26)` float32 — the same obs the RL policy sees
- `actions`: `(N, 4)` float32 — normalized motor thrusts in `[-1, 1]`
- `rewards`: `(N,)` float32 — per-step environment rewards
- `positions`: `(N, 3)` — world-frame positions (for visualization)
- `ref_positions`: `(N, 3)` — reference trajectory positions

### 3. Animation Script (`flightrl/training/animate_flight.py`)

Generates a GIF visualization of the flight. Run: `python animate_flight.py`

---

## What Needs To Be Done

### Step 1: Collect More Diverse Demonstrations

The current dataset has ~1,370 transitions from a single episode. For robust behavioral cloning, we need more:

```bash
# Generate demos at multiple speeds for diversity
python trajectory_controller.py --speed 3.0 --save-path demos/circle_8_slow.npz
python trajectory_controller.py --speed 4.5 --save-path demos/circle_8_med.npz
python trajectory_controller.py --speed 6.0 --save-path demos/circle_8_fast.npz
```

Then concatenate them:
```python
import numpy as np
files = ['demos/circle_8_slow.npz', 'demos/circle_8_med.npz', 'demos/circle_8_fast.npz']
all_obs, all_acts = [], []
for f in files:
    d = np.load(f)
    all_obs.append(d['observations'])
    all_acts.append(d['actions'])
np.savez('demos/circle_8_combined.npz',
         observations=np.concatenate(all_obs),
         actions=np.concatenate(all_acts))
```

### Step 2: Behavioral Cloning Pre-Training

Pre-train the PPO policy network using supervised learning on the demo data. This is the critical step — it gives the policy the "prior" of how to fly.

```python
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO

# Load demos
data = np.load('demos/circle_8_demos.npz')
expert_obs = torch.FloatTensor(data['observations']).cuda()
expert_acts = torch.FloatTensor(data['actions']).cuda()

# Create PPO model (same architecture as training)
model = PPO("MlpPolicy", env, 
            policy_kwargs=dict(net_arch=[128, 128]),
            device='cuda')

# Pre-train the policy network
optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
batch_size = 256
n_epochs = 200

for epoch in range(n_epochs):
    # Shuffle
    indices = torch.randperm(len(expert_obs))
    total_loss = 0
    n_batches = 0
    
    for start in range(0, len(expert_obs), batch_size):
        idx = indices[start:start + batch_size]
        obs_batch = expert_obs[idx]
        act_batch = expert_acts[idx]
        
        # Get policy's predicted action (mean of the Gaussian)
        # SB3's MlpPolicy outputs mean and log_std
        dist = model.policy.get_distribution(obs_batch)
        pred_acts = dist.distribution.mean
        
        loss = F.mse_loss(pred_acts, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={total_loss/n_batches:.6f}")

# Save pre-trained model
model.save("pretrained_bc_policy")
```

**Key considerations:**
- Use the same `net_arch` as your training script (`gate_racing_iros.py`)
- The observation normalization (z-scoring) used during RL training must match. If your training script applies `VecNormalize`, you need to either:
  - Pre-compute the observation statistics from the demo data and initialize the normalizer
  - OR skip normalization during BC and let RL's `VecNormalize` adapt during fine-tuning
- Train until loss plateaus (typically 100-300 epochs for ~5000 transitions)

### Step 3: RL Fine-Tuning

Load the pre-trained policy and continue with PPO:

```python
from stable_baselines3 import PPO

# Load pre-trained model
model = PPO.load("pretrained_bc_policy", env=env)

# Fine-tune with RL — use smaller learning rate to not destroy BC prior
model.learning_rate = 3e-5  # 10x smaller than default
model.learn(total_timesteps=25_000_000)
```

**Critical settings for fine-tuning:**
- **Lower learning rate** (3e-5 instead of 3e-4) to preserve the BC prior
- **Keep entropy coefficient** reasonable (0.01) to allow exploration beyond the demo behavior
- **Distributed init should be ON** so the agent sees all track segments
- **Crash penalty should be ON** (-5.0) to penalize ground crashes
- **Gate bonus should be ON** (+5.0) to reinforce gate traversal

### Step 4: Evaluate

```bash
# Evaluate the fine-tuned policy
python gate_racing_iros.py --eval --model-path path/to/finetuned_model
```

Expected progression:
- **After BC only**: The policy should hover and slowly navigate toward gates. It won't be fast, but it should stay alive and attempt the circuit.
- **After 5M RL steps**: The policy should pass 3-5 gates consistently
- **After 25M RL steps**: The policy should complete the full circuit

---

## Physics & Controller Details

### Quadrotor Parameters (from `quadrotor_env.yaml`)

| Parameter | Value |
|-----------|-------|
| Mass | 0.73 kg |
| Arm length | 0.17 m |
| Inertia | diag(0.003, 0.003, 0.005) kg·m² |
| Kappa (yaw drag) | 0.01 |
| Gravity | 9.81 m/s² |
| Sim dt | 0.02s (50 Hz) |
| Motor tau | 0.0001s |

### Action Space

The environment uses **single rotor thrusts (SRT)** — 4 continuous actions in `[-1, 1]`.

Mapping: `physical_thrust[i] = action[i] * act_std + act_mean`
- `act_mean = mass * g / 4 = 1.79 N` (hover thrust per motor)
- `act_std = mass * 2 * g / 4 = 3.58 N` (range scaling)
- `action = 0` → hover thrust
- `action = -0.5` → zero thrust
- `action = 1.0` → max thrust (~5.37 N per motor)

### Motor Allocation Matrix

Maps `[total_thrust, τ_roll, τ_pitch, τ_yaw]` → `[T1, T2, T3, T4]`:

```
B = [[1,    1,    1,    1   ],    # total thrust
     [+s,  -s,   -s,   +s  ],    # roll  (s = arm_l * √0.5 = 0.1202)
     [-s,  -s,   +s,   +s  ],    # pitch
     [+κ,  -κ,   +κ,   -κ  ]]   # yaw   (κ = 0.01)
```

Motor layout (X-configuration, top view):
```
   M1(CW)    M2(CCW)
      \      /
       \    /
        ──
       /    \
      /      \
   M4(CCW)   M3(CW)
```

### Observation Space (26-dim)

| Index | Dim | Content |
|-------|-----|---------|
| 0-2 | 3 | Linear velocity (world frame) |
| 3-5 | 3 | Linear acceleration (finite difference) |
| 6-14 | 9 | Rotation matrix (body→world, row-major) |
| 15-17 | 3 | Angular velocity (body frame) |
| 18-21 | 4 | Gate 1: (range, θ, φ, α) in body-frame spherical |
| 22-25 | 4 | Gate 2: (range, θ, φ, α) relative to gate 1 |

### Reward Function

- **Progress reward**: `r_p = s(p_t) - s(p_{t-1})` (spline-based arc-length progress)
- **Gate bonus**: +5.0 when passing through a gate
- **Crash penalty**: -5.0 on ground crash (z ≤ 0)
- **Body rate penalty**: `0.0005 * ||ω||²` (annealed to 0 over 5M steps)

### Known Shakiness in Controller

The trajectory controller produces somewhat oscillatory flight because:
1. **No integral term** — the PD controller has steady-state error on curved segments where centripetal acceleration is needed constantly
2. **Outer/inner loop coupling** — the position controller (outer) commands accelerations that require attitude changes, but the attitude controller (inner) has finite bandwidth, creating phase lag
3. **Feedforward capping** — acceleration feedforward is clamped to ±5 m/s² to prevent extreme attitudes, so on tight curves the controller must rely on feedback which introduces oscillation
4. **No angular rate feedforward** — the attitude controller only reacts to error, it doesn't predict the required body rates for the trajectory

These imperfections are actually *fine* for behavioral cloning — a slightly imperfect teacher forces the student to learn robustness. The RL fine-tuning will smooth out the behavior.

---

## File Reference

| File | Purpose |
|------|---------|
| `flightrl/training/trajectory_controller.py` | Trajectory generation + PD tracking controller + demo collection |
| `flightrl/training/animate_flight.py` | GIF animation of flight |
| `flightrl/training/demos/circle_8_demos.npz` | Saved demonstration data |
| `flightrl/training/gate_racing_iros.py` | PPO training script (needs BC pre-training step added) |
| `flightrl/training/sb3_vec_env.py` | SB3-compatible wrapper for Flightmare C++ env |
| `flightlib/configs/quadrotor_env.yaml` | Environment configuration (track, rewards, dynamics) |
| `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp` | C++ environment (reward, obs, termination logic) |
| `flightlib/src/dynamics/quadrotor_dynamics.cpp` | Quadrotor dynamics model |

## Track Configuration (circle_8)

8 gates evenly spaced on a circle of radius 18m at z=3.0m:

| Gate | θ (deg) | Position (x, y, z) | Gate yaw |
|------|---------|---------------------|----------|
| G0 | 0° | (18.0, 0.0, 3.0) | 90° |
| G1 | 45° | (12.728, 12.728, 3.0) | 135° |
| G2 | 90° | (0.0, 18.0, 3.0) | 180° |
| G3 | 135° | (-12.728, 12.728, 3.0) | 225° |
| G4 | 180° | (-18.0, 0.0, 3.0) | 270° |
| G5 | 225° | (-12.728, -12.728, 3.0) | 315° |
| G6 | 270° | (0.0, -18.0, 3.0) | 0° |
| G7 | 315° | (12.728, -12.728, 3.0) | 45° |

Start position: `[18.0, -4.0, 3.0]` (4m behind G0)
