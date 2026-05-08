#!/usr/bin/env python3
"""
Stream a trained policy's rollouts to Unity for live rendering.

Workflow:
  1. On Mac: open Unity (Top_Level_Scene), enter the IP of the machine running
     this script, click CONNECT, pick a scene.
  2. On the env host: apply the desired track and run this script.
  3. Screen-record Unity's Game window (e.g. QuickTime → New Screen Recording).

Usage:
  python viz_unity.py --model saved/curriculum_v2/E_circle_8_v4/best_model.zip \
                      --vecnorm saved/curriculum_v2/E_circle_8_v4/vec_normalize.pkl \
                      --scene-id 0 -n 5

  --scene-id  0=Industrial 1=Warehouse 2=Garage 3=Forest 4=Tunnels
  -n          number of episodes to stream

Make sure vec_env.yaml has `render: yes` (already set in this repo) so the
C++ side opens its ZMQ pub/sub on ports 10253/10254.
"""

import argparse
import os
import shutil
import time
import numpy as np

VEC_YAML = os.path.expanduser("~/flightmare/flightlib/configs/vec_env.yaml")
VEC_YAML_BACKUP = VEC_YAML + ".viz_backup"


def patch_vec_yaml_for_unity():
    """Swap num_envs → 1 and render → yes for a clean Unity demo. Revert later."""
    if not os.path.exists(VEC_YAML_BACKUP):
        shutil.copy(VEC_YAML, VEC_YAML_BACKUP)
    with open(VEC_YAML) as f:
        text = f.read()
    new = []
    for line in text.splitlines():
        if line.strip().startswith("num_envs:"):
            new.append("  num_envs: 1")
        elif line.strip().startswith("render:"):
            new.append("  render: yes")
        else:
            new.append(line)
    with open(VEC_YAML, "w") as f:
        f.write("\n".join(new) + "\n")


def restore_vec_yaml():
    if os.path.exists(VEC_YAML_BACKUP):
        shutil.move(VEC_YAML_BACKUP, VEC_YAML)


import flightgym
from sb3_vec_env import FlightmareSB3VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--scene-id", type=int, default=0,
                    help="0=Industrial 1=Warehouse 2=Garage 3=Forest 4=Tunnels")
    ap.add_argument("-n", "--n-episodes", type=int, default=5)
    ap.add_argument("--stochastic", action="store_true",
                    help="Use stochastic actions (default deterministic).")
    ap.add_argument("--realtime", action="store_true", default=True,
                    help="Sleep between steps so Unity renders at sim_dt rate.")
    args = ap.parse_args()

    patch_vec_yaml_for_unity()

    # Build env (render flag picked up from vec_env.yaml -> env.render).
    cpp_env = flightgym.QuadrotorEnv_v1()
    raw_env = FlightmareSB3VecEnv(cpp_env)

    # Eval should always start at start_pos, not distributed init.
    raw_env.set_random_reset(False)

    env = VecNormalize.load(args.vecnorm, raw_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model, env=env, device="cpu")

    # Tell C++ to set the requested scene and connect to a running Unity.
    print(f"Connecting to Unity (scene_id={args.scene_id}). Make sure Unity "
          f"is running and clicked CONNECT to this host's IP...")
    if not cpp_env.connectUnity():
        print("ERROR: Unity connection timed out. Is Unity running?")
        return
    print("Connected. Streaming episodes...")

    sim_dt = 0.02  # matches quadrotor_env.yaml; sleep target between steps
    obs = env.reset()
    n_envs = env.num_envs
    finished = 0
    ep_rewards = np.zeros(n_envs)
    ep_lengths = np.zeros(n_envs, dtype=int)

    try:
        while finished < args.n_episodes:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rew, done, info = env.step(action)
            ep_rewards += rew
            ep_lengths += 1

            for i in range(n_envs):
                if done[i] and finished < args.n_episodes:
                    finished += 1
                    print(f"  ep {finished}/{args.n_episodes}: env={i} "
                          f"R={ep_rewards[i]:.2f} len={ep_lengths[i]}")
                    ep_rewards[i] = 0
                    ep_lengths[i] = 0

            if args.realtime:
                # Unity renders at the rate frames arrive; this paces it
                # to wall-clock real time so the demo is not 50× sped up.
                dt = time.time() - t0
                if dt < sim_dt:
                    time.sleep(sim_dt - dt)
    finally:
        cpp_env.disconnectUnity()
        restore_vec_yaml()
        print("Disconnected from Unity. vec_env.yaml restored.")


if __name__ == "__main__":
    main()
