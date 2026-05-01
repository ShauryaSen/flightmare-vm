#!/usr/bin/env python3
"""Sanity check for the new 12-gate track (G1 reorientation + inserted gate)."""
import numpy as np
import flightgym

env = flightgym.QuadrotorEnv_v1()
n = env.getNumOfEnvs()
od, ad = env.getObsDim(), env.getActDim()
extras = list(env.getExtraInfoNames())

print(f"obs_dim={od}  act_dim={ad}  num_envs={n}  extras={len(extras)}")
assert od == 26, f"expected 26-dim obs, got {od}"

env.setRandomReset(False)

obs = np.zeros([n, od], dtype=np.float32)
act = np.zeros([n, ad], dtype=np.float32)
rew = np.zeros(n, dtype=np.float32)
done = np.zeros(n, dtype=bool)
info = np.zeros([n, len(extras)], dtype=np.float32)

env.reset(obs)

env.step(act, obs, rew, done, info)
gate_idx_col = extras.index("gate_idx")
pos_cols = [extras.index(c) for c in ("pos_x", "pos_y", "pos_z")]
vel_norm_col = extras.index("vel_norm")
r_prog_col = extras.index("r_progress")

print(f"\nAfter reset+1 step (random_reset=False):")
print(f"  pos env0   = [{info[0, pos_cols[0]]:+7.3f}, {info[0, pos_cols[1]]:+7.3f}, {info[0, pos_cols[2]]:+7.3f}]")
print(f"  pos env50  = [{info[50, pos_cols[0]]:+7.3f}, {info[50, pos_cols[1]]:+7.3f}, {info[50, pos_cols[2]]:+7.3f}]")
print(f"  gate_idx   : {set(int(x) for x in info[:, gate_idx_col])}")
print(f"  vel_norm   : mean={info[:, vel_norm_col].mean():.3f}  max={info[:, vel_norm_col].max():.3f}")

unique_idx = set(int(x) for x in info[:, gate_idx_col])
assert unique_idx == {0}, f"random_reset=False must spawn at gate 0, got {unique_idx}"

start_x = info[0, pos_cols[0]]
start_y = info[0, pos_cols[1]]
start_z = info[0, pos_cols[2]]
assert abs(start_x - 17.4) < 0.5 and abs(start_y - (-5.0)) < 0.5 and abs(start_z - 6.31) < 0.5, \
    f"start position drift: ({start_x:.3f}, {start_y:.3f}, {start_z:.3f}) vs expected (17.4, -5, 6.31)"

print("\nRunning 200 random-action steps...")
total_rew = np.zeros(n, dtype=np.float32)
for _ in range(200):
    act[:] = np.random.uniform(-0.3, 0.3, (n, ad)).astype(np.float32)
    env.step(act, obs, rew, done, info)
    total_rew += rew

print(f"  reward range : [{rew.min():+.4f}, {rew.max():+.4f}]")
print(f"  cumulative   : mean={total_rew.mean():+.3f}  range=[{total_rew.min():+.3f}, {total_rew.max():+.3f}]")
print(f"  end gate_idx : {sorted(set(int(x) for x in info[:, gate_idx_col]))}")
print(f"  end vel_norm : mean={info[:, vel_norm_col].mean():.3f}  max={info[:, vel_norm_col].max():.3f}")

assert rew.std() > 0 or total_rew.std() > 0, "no reward variation across envs — step may not be running"

print("\nOK: env loads, steps, computes rewards, advances state.")
