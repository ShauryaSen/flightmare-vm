#!/usr/bin/env python3
"""Condition C trainer — automated random-track curriculum (IROS21 §III.D).

Each episode samples a fresh track from a TrackGenerator (gate primitives
in the previous gate's frame). Complexity c ∈ [0, 1] starts at 0 (straight
line — linear_8.yaml baseline) and is bumped by an AdaptiveComplexity
controller using the completion ratio across all parallel envs.

The C++ vec_env auto-resets done envs BEFORE the callback fires, so when we
call applyPerturbedGates(i) the env is already at a freshly-sampled init
pose. applyPerturbedGates internally resets s_prev_, so the next step's
spline-progress delta is well-defined against the new track.

Caveat: SB3 calls the callback after the WHOLE batch step (all envs), not
per-env. The first observation of the next episode (returned by the C++
auto-reset already inside step()) reflects the OLD track. The very first
post-perturbation step's progress delta is computed against the new spline
starting from start_pos, which is correct because applyPerturbedGates
recomputes s_prev_ at quad_state_.p. Acceptable: one slightly stale obs out
of millions.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
FLIGHTMARE_PATH = os.environ.get("FLIGHTMARE_PATH",
                                 str(THIS_DIR.parents[2]))
TRACKS_DIR = Path(FLIGHTMARE_PATH) / "flightlib" / "configs" / "tracks"

sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(TRAINING_DIR / "hp_sweep"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import flightgym
from sb3_vec_env import FlightmareSB3VecEnv

from track_generator import TrackGenerator
from adaptive_complexity import AdaptiveComplexity

ACTIVATION_MAP = {"Tanh": nn.Tanh, "ReLU": nn.ReLU}


class RandomTrackCallback(BaseCallback):
    """Per-step callback that mutates env-i's gates on done[i] and updates
    the complexity controller using the completion ratio."""

    def __init__(self, raw_env, generator: TrackGenerator,
                 controller: AdaptiveComplexity, run_dir: str,
                 log_every: int = 100_000, verbose: int = 1):
        super().__init__(verbose)
        self.raw_env = raw_env  # FlightmareSB3VecEnv with .wrapper exposed
        self.generator = generator
        self.controller = controller
        self.run_dir = run_dir
        self.log_every = log_every

        # mirror ProgressCallback bookkeeping for the run-level progress log
        self._ep_rewards = np.zeros(0)
        self._ep_lengths = np.zeros(0, dtype=int)
        self._ep_max_gates = np.zeros(0)
        self._completed_rewards = []
        self._completed_lengths = []
        self._completed_max_gates = []
        self._initialized = False
        self._next_log = log_every
        self.progress = []
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=int)
        self._ep_max_gates = np.zeros(n)
        self._initialized = True
        # Verify the C++ wrapper handle is reachable.
        assert hasattr(self.raw_env, "wrapper"), \
            "FlightmareSB3VecEnv must expose .wrapper for applyPerturbedGates"

    def _perturb_env(self, env_id: int):
        pos, yaw = self.generator.sample(self.controller.complexity)
        # MatrixRowMajor<float>; pybind expects np.float32 with shape (N, 3).
        self.raw_env.wrapper.applyPerturbedGates(env_id,
                                                 pos.astype(np.float32),
                                                 yaw.astype(np.float32))

    def _on_step(self) -> bool:
        if not self._initialized:
            return True
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        self._ep_rewards += rewards
        self._ep_lengths += 1

        for i in range(len(dones)):
            gi = float(infos[i].get("gate_idx", 0.0))
            if gi > self._ep_max_gates[i]:
                self._ep_max_gates[i] = gi
            if dones[i]:
                # `gates_passed` is cumulative within the episode (does NOT
                # wrap), so >= num_gates means the drone completed the
                # track. `gate_idx` wraps and is a worse signal here.
                gates_passed = float(infos[i].get("gates_passed",
                                     infos[i].get("gate_idx", 0.0)))
                self._completed_rewards.append(float(self._ep_rewards[i]))
                self._completed_lengths.append(int(self._ep_lengths[i]))
                self._completed_max_gates.append(float(self._ep_max_gates[i]))
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_max_gates[i] = 0.0

                # Push to controller (may bump complexity at update_every).
                self.controller.push_episode(gates_passed,
                                             current_step=self.num_timesteps)
                # Now resample track for THIS env.
                try:
                    self._perturb_env(i)
                except Exception as e:
                    print(f"[rand_track_cb] applyPerturbedGates failed env={i}: {e!r}",
                          flush=True)

        if self.num_timesteps >= self._next_log:
            self._log_snapshot()
            self._next_log += self.log_every
        return True

    def _log_snapshot(self):
        if not self._completed_rewards:
            return
        recent_n = min(100, len(self._completed_rewards))
        snap = {
            "step": int(self.num_timesteps),
            "ep_rew_mean": float(np.mean(self._completed_rewards[-recent_n:])),
            "ep_len_mean": float(np.mean(self._completed_lengths[-recent_n:])),
            "ep_max_gate_mean": float(np.mean(self._completed_max_gates[-recent_n:])),
            "ep_max_gate_max": float(np.max(self._completed_max_gates[-recent_n:])),
            "n_episodes": len(self._completed_rewards),
            "complexity": float(self.controller.complexity),
            "total_episodes": int(self.controller.total_episodes),
        }
        self.progress.append(snap)
        with open(os.path.join(self.run_dir, "progress.json"), "w") as f:
            json.dump(self.progress, f, indent=2)
        with open(os.path.join(self.run_dir, "complexity_history.json"), "w") as f:
            json.dump(self.controller.history, f, indent=2)

        if snap["ep_rew_mean"] > self.best_mean_reward:
            self.best_mean_reward = snap["ep_rew_mean"]
            best_dir = os.path.join(self.run_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save(os.path.join(best_dir, "best_model"))
            env = self.training_env
            if isinstance(env, VecNormalize):
                env.save(os.path.join(best_dir, "vec_normalize.pkl"))
            if self.verbose:
                print(f"  ★ step={snap['step']:>10,}  rew={snap['ep_rew_mean']:7.2f}  "
                      f"len={snap['ep_len_mean']:6.0f}  gate={snap['ep_max_gate_mean']:.2f}  "
                      f"c={snap['complexity']:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("hparams_json", type=str)
    p.add_argument("--total-timesteps", type=int, default=100_000_000)
    p.add_argument("--log-every", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-model", type=str, default=None)
    p.add_argument("--load-vecnorm", type=str, default=None)
    p.add_argument("--c-init", type=float, default=0.0)
    p.add_argument("--c-max", type=float, default=1.0)
    p.add_argument("--c-step", type=float, default=0.05)
    p.add_argument("--c-update-every", type=int, default=100,
                   help="Episodes between complexity updates (lower = faster smoke).")
    p.add_argument("--baseline-yaml", type=str,
                   default=str(TRACKS_DIR / "linear_8.yaml"))
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    with open(args.hparams_json) as f:
        hp = json.load(f)

    raw_env = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
    raw_env.seed(args.seed)

    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        env = VecNormalize.load(args.load_vecnorm, raw_env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    activation = ACTIVATION_MAP.get(hp.get("activation_fn", "Tanh"), nn.Tanh)
    net_arch = hp.get("net_arch", [256, 256])
    n_steps = int(hp.get("n_steps", 500))
    rollout_size = n_steps * env.num_envs
    batch_size = min(int(hp.get("batch_size", rollout_size)), rollout_size)
    if rollout_size % batch_size != 0:
        batch_size = rollout_size

    if args.load_model and os.path.exists(args.load_model):
        from stable_baselines3.common.buffers import RolloutBuffer
        model = PPO.load(args.load_model, env=env, device="cpu",
                         tensorboard_log=os.path.join(args.run_dir, "tb"))
        model.learning_rate = float(hp.get("learning_rate", 3e-4))
        model.lr_schedule = lambda _: float(hp.get("learning_rate", 3e-4))
        for g in model.policy.optimizer.param_groups:
            g["lr"] = float(hp.get("learning_rate", 3e-4))
        model.ent_coef = float(hp.get("ent_coef", 0.01))
        model.clip_range = lambda _: float(hp.get("clip_range", 0.2))
        model.gamma = float(hp.get("gamma", 0.99))
        model.gae_lambda = float(hp.get("gae_lambda", 0.95))
        model.vf_coef = float(hp.get("vf_coef", 0.5))
        model.max_grad_norm = float(hp.get("max_grad_norm", 0.5))
        model.n_epochs = int(hp.get("n_epochs", 10))
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.rollout_buffer = RolloutBuffer(
            n_steps, model.observation_space, model.action_space,
            device=model.device, gamma=model.gamma,
            gae_lambda=model.gae_lambda, n_envs=model.n_envs)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=float(hp.get("learning_rate", 3e-4)),
            n_steps=n_steps, batch_size=batch_size,
            n_epochs=int(hp.get("n_epochs", 10)),
            gamma=float(hp.get("gamma", 0.99)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            clip_range=float(hp.get("clip_range", 0.2)),
            ent_coef=float(hp.get("ent_coef", 0.01)),
            vf_coef=float(hp.get("vf_coef", 0.5)),
            max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
            policy_kwargs=dict(activation_fn=activation,
                               net_arch=dict(pi=list(net_arch), vf=list(net_arch))),
            verbose=0, tensorboard_log=os.path.join(args.run_dir, "tb"),
            device="cpu", seed=args.seed,
        )

    generator = TrackGenerator.from_baseline_yaml(args.baseline_yaml,
                                                  seed=args.seed)
    controller = AdaptiveComplexity(
        num_gates=generator.num_gates,
        c_init=args.c_init,
        c_max=args.c_max,
        step_up=args.c_step,
        step_down=args.c_step,
        update_every=args.c_update_every,
    )

    cb = RandomTrackCallback(raw_env, generator, controller,
                             args.run_dir, log_every=args.log_every, verbose=1)

    t0 = time.time()
    print(f"[rand_track] start {args.run_dir}  steps={args.total_timesteps:,}  "
          f"c_init={args.c_init}  baseline={args.baseline_yaml}", flush=True)

    train_error = None
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=cb,
                    progress_bar=False, reset_num_timesteps=True)
    except Exception as e:
        train_error = repr(e)
        print(f"[rand_track] ERROR during learn(): {train_error}", flush=True)

    train_seconds = time.time() - t0
    model.save(os.path.join(args.run_dir, "final_model"))
    env.save(os.path.join(args.run_dir, "final_vec_normalize.pkl"))
    cb._log_snapshot()

    final_progress = cb.progress[-1] if cb.progress else {}
    result = {
        "run_dir": args.run_dir,
        "hparams": hp,
        "total_timesteps": args.total_timesteps,
        "train_seconds": train_seconds,
        "train_error": train_error,
        "final_ep_rew_mean": final_progress.get("ep_rew_mean", float("nan")),
        "final_ep_len_mean": final_progress.get("ep_len_mean", 0),
        "final_ep_max_gate_mean": final_progress.get("ep_max_gate_mean", 0),
        "best_mean_reward": cb.best_mean_reward
            if cb.best_mean_reward != -np.inf else float("nan"),
        "final_complexity": float(controller.complexity),
        "total_episodes": int(controller.total_episodes),
    }
    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[rand_track] done {args.run_dir} in {train_seconds:.0f}s  "
          f"c={result['final_complexity']:.2f}  "
          f"best_rew={result['best_mean_reward']:.2f}", flush=True)


if __name__ == "__main__":
    main()
