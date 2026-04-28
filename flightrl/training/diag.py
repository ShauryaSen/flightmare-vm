#!/usr/bin/env python3
"""Quick diagnostic — verify env setup before training.

Usage: python diag.py
"""
import numpy as np
import flightgym
import time

env = flightgym.QuadrotorEnv_v1()
n = env.getNumOfEnvs()
od, ad = env.getObsDim(), env.getActDim()

print(f"{'='*50}")
print(f"  Flightmare Environment Diagnostic")
print(f"{'='*50}")
print(f"  Obs dim:  {od}")
print(f"  Act dim:  {ad}")
print(f"  Num envs: {n}")

obs = np.zeros([n, od], dtype=np.float32)
act = np.zeros([n, ad], dtype=np.float32)
rew = np.zeros(n, dtype=np.float32)
done = np.zeros(n, dtype=bool)
info = np.zeros([n, 0], dtype=np.float32)

env.reset(obs)
print(f"\nAfter reset (env 0):")
print(f"  pos:      [{obs[0,0]:+7.2f}, {obs[0,1]:+7.2f}, {obs[0,2]:+7.2f}]")
print(f"  euler:    [{obs[0,3]:+7.3f}, {obs[0,4]:+7.3f}, {obs[0,5]:+7.3f}]")
print(f"  vel:      [{obs[0,6]:+7.3f}, {obs[0,7]:+7.3f}, {obs[0,8]:+7.3f}]")
print(f"  omega:    [{obs[0,9]:+7.3f}, {obs[0,10]:+7.3f}, {obs[0,11]:+7.3f}]")
if od > 12:
    print(f"  gate_rel: [{obs[0,12]:+7.2f}, {obs[0,13]:+7.2f}, {obs[0,14]:+7.2f}]")

# Run 500 random steps to check dynamics + rewards
rewards = []
crash_count = 0
t0 = time.time()
for _ in range(500):
    act[:] = np.random.uniform(-1, 1, (n, ad)).astype(np.float32)
    env.step(act, obs, rew, done, info)
    rewards.append(rew.copy())
    crash_count += done.sum()
elapsed = time.time() - t0
total_steps = 500 * n

all_rew = np.array(rewards)
print(f"\n500 steps × {n} envs = {total_steps:,} total steps:")
print(f"  FPS:          {total_steps / elapsed:,.0f}")
print(f"  reward range: [{all_rew.min():.3f}, {all_rew.max():.3f}]")
print(f"  reward mean:  {all_rew.mean():.3f}")
print(f"  reward std:   {all_rew.std():.3f}")
print(f"  total crashes: {crash_count} / {total_steps} ({crash_count/total_steps*100:.1f}%)")

print(f"\nFinal state (env 0):")
print(f"  pos:      [{obs[0,0]:+7.2f}, {obs[0,1]:+7.2f}, {obs[0,2]:+7.2f}]")
if od > 12:
    print(f"  gate_rel: [{obs[0,12]:+7.2f}, {obs[0,13]:+7.2f}, {obs[0,14]:+7.2f}]")

print(f"\n{'='*50}")
print(f"  ✓ Diagnostic complete")
print(f"{'='*50}")
