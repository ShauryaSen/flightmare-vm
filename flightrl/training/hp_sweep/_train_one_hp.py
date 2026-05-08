#!/usr/bin/env python3
"""Single PPO hyperparameter trial. Invoked as a subprocess by the Optuna orchestrator.

Reads a hyperparameter JSON, trains PPO for `--total-timesteps`, evaluates
deterministically from start_pos, and writes result.json + progress.json.

Run as subprocess so each trial gets a fresh Python interpreter and the C++
env reloads cleanly. The orchestrator passes the same `flightlib/configs/
quadrotor_env.yaml` to all trials — only PPO hyperparameters vary.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, TRAINING_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


ACTIVATION_MAP = {"Tanh": nn.Tanh, "ReLU": nn.ReLU}


class ProgressCallback(BaseCallback):
    def __init__(self, run_dir, log_every=100_000, verbose=1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.log_every = log_every
        self.best_mean_reward = -np.inf
        self._ep_rewards = np.zeros(0)
        self._ep_lengths = np.zeros(0, dtype=int)
        self._ep_max_gates = np.zeros(0)
        self._completed_rewards = []
        self._completed_lengths = []
        self._completed_max_gates = []
        self._initialized = False
        self._next_log = log_every
        self.progress = []

    def _init_callback(self):
        n = self.training_env.num_envs
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=int)
        self._ep_max_gates = np.zeros(n)
        self._initialized = True

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
                self._completed_rewards.append(float(self._ep_rewards[i]))
                self._completed_lengths.append(int(self._ep_lengths[i]))
                self._completed_max_gates.append(float(self._ep_max_gates[i]))
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_max_gates[i] = 0.0
        if self.num_timesteps >= self._next_log:
            self._log_snapshot()
            self._next_log += self.log_every
        return True

    def _log_snapshot(self):
        if len(self._completed_rewards) == 0:
            return
        recent_n = min(100, len(self._completed_rewards))
        snap = {
            "step": int(self.num_timesteps),
            "ep_rew_mean": float(np.mean(self._completed_rewards[-recent_n:])),
            "ep_len_mean": float(np.mean(self._completed_lengths[-recent_n:])),
            "ep_max_gate_mean": float(np.mean(self._completed_max_gates[-recent_n:])),
            "ep_max_gate_max": float(np.max(self._completed_max_gates[-recent_n:])),
            "n_episodes": len(self._completed_rewards),
        }
        self.progress.append(snap)
        with open(os.path.join(self.run_dir, "progress.json"), "w") as f:
            json.dump(self.progress, f, indent=2)
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
                      f"len={snap['ep_len_mean']:6.0f}  gate={snap['ep_max_gate_mean']:.2f}",
                      flush=True)


def evaluate(model, raw_env, vecnorm_env, n_episodes=10, max_steps=1500):
    raw_env.set_random_reset(False)
    vecnorm_env.training = False
    vecnorm_env.norm_reward = False
    obs = vecnorm_env.reset()
    n_envs = vecnorm_env.num_envs
    cur_rewards = np.zeros(n_envs)
    cur_lengths = np.zeros(n_envs, dtype=int)
    cur_max_gates = np.zeros(n_envs)
    ep_rewards, ep_lengths, ep_max_gates = [], [], []
    completed = 0
    total_steps = 0
    while completed < n_episodes and total_steps < max_steps * 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vecnorm_env.step(action)
        cur_rewards += reward
        cur_lengths += 1
        for i in range(n_envs):
            gi = float(infos[i].get("gate_idx", 0.0))
            if gi > cur_max_gates[i]:
                cur_max_gates[i] = gi
            if done[i]:
                ep_rewards.append(float(cur_rewards[i]))
                ep_lengths.append(int(cur_lengths[i]))
                ep_max_gates.append(float(cur_max_gates[i]))
                cur_rewards[i] = 0.0
                cur_lengths[i] = 0
                cur_max_gates[i] = 0.0
                completed += 1
                if completed >= n_episodes:
                    break
        total_steps += 1
    if not ep_rewards:
        return {"eval_completed": 0, "eval_mean_reward": float("nan"),
                "eval_mean_length": 0, "eval_mean_max_gate": 0,
                "eval_max_gate_idx": 0}
    return {
        "eval_completed": len(ep_rewards),
        "eval_mean_reward": float(np.mean(ep_rewards)),
        "eval_mean_length": float(np.mean(ep_lengths)),
        "eval_mean_max_gate": float(np.mean(ep_max_gates)),
        "eval_max_gate_idx": float(max(ep_max_gates)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("hparams_json", type=str)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Optional warm-start model .zip")
    parser.add_argument("--load-vecnorm", type=str, default=None,
                        help="Optional warm-start VecNormalize .pkl")
    args = parser.parse_args()

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
    n_steps = int(hp.get("n_steps", 750))
    num_envs = env.num_envs
    rollout_size = n_steps * num_envs
    batch_size = min(int(hp.get("batch_size", rollout_size)), rollout_size)
    if rollout_size % batch_size != 0:
        batch_size = rollout_size

    if args.load_model and os.path.exists(args.load_model):
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
        # Override rollout size + reallocate buffer so sampled n_steps
        # actually takes effect on the warm-started model. Without this,
        # n_steps stays at whatever value the loaded model was trained with.
        # net_arch and activation_fn are baked into the network architecture
        # at PPO.load() time and CANNOT be changed — sampled values are
        # ignored for warm-started trials (documented in the report).
        from stable_baselines3.common.buffers import RolloutBuffer
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.rollout_buffer = RolloutBuffer(
            n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=float(hp.get("learning_rate", 3e-4)),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(hp.get("n_epochs", 10)),
            gamma=float(hp.get("gamma", 0.99)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            clip_range=float(hp.get("clip_range", 0.2)),
            ent_coef=float(hp.get("ent_coef", 0.01)),
            vf_coef=float(hp.get("vf_coef", 0.5)),
            max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
            policy_kwargs=dict(activation_fn=activation,
                               net_arch=dict(pi=list(net_arch), vf=list(net_arch))),
            verbose=0,
            tensorboard_log=os.path.join(args.run_dir, "tb"),
            device="cpu",
            seed=args.seed,
        )

    cb = ProgressCallback(args.run_dir, log_every=args.log_every, verbose=1)
    t0 = time.time()
    print(f"[trial] starting {args.run_dir}  steps={args.total_timesteps:,}  hp={hp}", flush=True)

    train_error = None
    try:
        # Always reset_num_timesteps=True. SB3 treats total_timesteps as an
        # absolute cap when reset_num_timesteps=False, so warm-started trials
        # would only train (total_timesteps - warm_start_steps) effective
        # steps — which can be near zero. Resetting gives the trial the full
        # intended budget. The optimizer state from the warm-start is still
        # preserved (PPO.load loads the optimizer); only the step counter
        # and LR/anneal schedules restart, which is what we want for a
        # fine-tune trial.
        model.learn(total_timesteps=args.total_timesteps, callback=cb,
                    progress_bar=False,
                    reset_num_timesteps=True)
    except Exception as e:
        train_error = repr(e)
        print(f"[trial] ERROR during learn(): {train_error}", flush=True)

    train_seconds = time.time() - t0
    model.save(os.path.join(args.run_dir, "final_model"))
    env.save(os.path.join(args.run_dir, "final_vec_normalize.pkl"))
    cb._log_snapshot()

    eval_metrics = {}
    eval_error = None
    try:
        best_dir = os.path.join(args.run_dir, "best_model")
        if os.path.exists(os.path.join(best_dir, "vec_normalize.pkl")):
            eval_raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
            eval_env = VecNormalize.load(
                os.path.join(best_dir, "vec_normalize.pkl"), eval_raw)
            best_model = PPO.load(os.path.join(best_dir, "best_model"))
            eval_metrics = evaluate(best_model, eval_raw, eval_env)
            eval_raw.close()
        else:
            eval_metrics = evaluate(model, raw_env, env)
    except Exception as e:
        eval_error = repr(e)
        eval_metrics = {"eval_error": eval_error}

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
        **eval_metrics,
    }
    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[trial] done {args.run_dir} in {train_seconds:.0f}s  "
          f"best_rew={result['best_mean_reward']:.2f}  "
          f"eval_max_gate={result.get('eval_max_gate_idx', 0)}", flush=True)


if __name__ == "__main__":
    main()
