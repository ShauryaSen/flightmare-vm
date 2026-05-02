#!/usr/bin/env python3
"""
Behavioral Cloning Pre-Training for Drone Racing RL Warm-Start.

Trains the same MLP policy that PPO uses (256x256 Tanh) via supervised MSE
on (obs, action) pairs from a classical trajectory-tracking controller, then
saves it as a PPO-loadable .zip plus a VecNormalize.pkl with obs stats
computed from the demo data.

The output is consumed by gate_racing_iros.py via:
  python gate_racing_iros.py --train 1 \
    --load-model saved/bc_pretrained/bc_model.zip \
    --load-vecnorm saved/bc_pretrained/vec_normalize.pkl

Usage:
  python train_bc.py --demos demos/circle_8_slow.npz demos/circle_8_med.npz \
    --save-dir saved/bc_pretrained --epochs 200 --batch-size 256 --lr 1e-3
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flightgym
from sb3_vec_env import FlightmareSB3VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def load_demos(paths, gamma=0.99):
    """Load (obs, act, returns) per file. Returns are computed PER FILE so
    they don't bleed across episode boundaries (each demo file is one
    rollout from start_pos). Bootstrap=0 at the file end (terminal)."""
    obs_list, act_list, ret_list = [], [], []
    total_reward = 0.0
    for p in paths:
        d = np.load(p)
        obs_p = d["observations"].astype(np.float32)
        act_p = d["actions"].astype(np.float32)
        if "rewards" in d:
            r = d["rewards"].astype(np.float32)
            total_reward += float(r.sum())
        else:
            r = np.zeros(len(obs_p), dtype=np.float32)
        # Discounted Monte-Carlo returns: G_t = r_t + γ G_{t+1}, G_T = 0
        ret_p = np.zeros_like(r)
        running = 0.0
        for t in range(len(r) - 1, -1, -1):
            running = r[t] + gamma * running
            ret_p[t] = running
        obs_list.append(obs_p)
        act_list.append(act_p)
        ret_list.append(ret_p)
        print(f"  {p}: {len(obs_p)} transitions"
              + (f", reward_sum={float(r.sum()):.1f}"
                 f", return_max={ret_p.max():.1f}"
                 f", return_mean={ret_p.mean():.1f}"))
    obs = np.concatenate(obs_list).astype(np.float32)
    acts = np.concatenate(act_list).astype(np.float32)
    rets = np.concatenate(ret_list).astype(np.float32)
    print(f"Total demos: {len(obs)} transitions, total_reward={total_reward:.1f}")
    return obs, acts, rets


def policy_forward(policy, obs):
    """Forward pass returning (action_mean, value). Replicates the relevant
    parts of ActorCriticPolicy.forward() but skips action sampling."""
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    mean_actions = policy.action_net(latent_pi)
    values = policy.value_net(latent_vf).squeeze(-1)
    return mean_actions, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", nargs="+", required=True,
                        help="Paths to demo .npz files (concatenated).")
    parser.add_argument("--save-dir", default="saved/bc_pretrained")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of demos held out for validation MSE.")
    parser.add_argument("--input-noise-std", type=float, default=0.05,
                        help="Gaussian noise std added to normalized obs each "
                             "batch (DART-style robustness to state drift).")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for Monte Carlo return targets.")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Weight on value-function MSE in the joint loss.")
    parser.add_argument("--init-std", type=float, default=0.1,
                        help="Initial action stddev for PPO fine-tuning. "
                             "Default 0.1 keeps early stochastic actions "
                             "close to the BC mean; SB3 default is 1.0 "
                             "which destroys the BC prior.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true",
                        help="After BC, run the policy in env for one episode "
                             "and report gates_passed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load demos ─────────────────────────────────────────────────────
    print("\n[1/4] Loading demos")
    obs, acts, rets = load_demos(args.demos, gamma=args.gamma)
    n = len(obs)
    if n < 200:
        print(f"WARNING: only {n} transitions — BC will overfit.")

    # Train/val split
    perm = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    # ── Build env (needed to instantiate PPO with the right shapes) ────
    print("\n[2/4] Building env + VecNormalize")
    cpp_env = flightgym.QuadrotorEnv_v1()
    raw_env = FlightmareSB3VecEnv(cpp_env)
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Inject demo-derived running stats so the policy trains on z-scored obs
    # that match what gate_racing_iros.py will feed it after fine-tune resumes.
    obs_mean = obs.mean(axis=0)
    obs_var = obs.var(axis=0) + 1e-8
    env.obs_rms.mean = obs_mean.astype(np.float64)
    env.obs_rms.var = obs_var.astype(np.float64)
    env.obs_rms.count = float(n)
    print(f"  obs_rms: mean[0]={obs_mean[0]:.3f} std[0]={np.sqrt(obs_var[0]):.3f}"
          f"  (computed from {n} demo transitions)")

    # ── Create PPO model with same architecture as gate_racing_iros.py ──
    print("\n[3/4] Creating PPO model and BC-training the policy")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,  # PPO LR (used after fine-tune resume, not here)
        n_steps=1500, batch_size=150_000, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(activation_fn=nn.Tanh,
                           net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device=args.device, verbose=0, seed=args.seed,
    )

    # Apply VecNormalize-equivalent z-scoring + clip on the demo obs
    obs_norm = (obs - obs_mean) / np.sqrt(obs_var)
    obs_norm = np.clip(obs_norm, -10.0, 10.0).astype(np.float32)

    device = torch.device(args.device)
    obs_t = torch.from_numpy(obs_norm).to(device)
    act_t = torch.from_numpy(acts).to(device)
    ret_t = torch.from_numpy(rets).to(device)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)

    print(f"  arch=pi:[256,256] act=Tanh  device={args.device}")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr}"
          f"  train={len(tr_idx)} val={len(val_idx)} vf_coef={args.vf_coef}")
    print(f"  return targets: mean={rets.mean():.2f} max={rets.max():.2f}")

    # Initial val loss
    with torch.no_grad():
        mean_a, val_v = policy_forward(model.policy, obs_t[val_idx])
        init_act_mse = F.mse_loss(mean_a, act_t[val_idx]).item()
        init_v_mse = F.mse_loss(val_v, ret_t[val_idx]).item()
    print(f"  epoch    0  act_mse=    --  val_act={init_act_mse:.6f} "
          f" val_vf={init_v_mse:.4f}  (init)")

    # Early stopping: keep the weights from the epoch with the lowest val loss.
    # BC overfits past ~30 epochs on small demo datasets; deploying the
    # latest weights yields a worse policy than the best-val checkpoint.
    import copy
    best_val = float("inf")
    best_state = copy.deepcopy(model.policy.state_dict())
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        # Shuffle train indices
        rng_perm = torch.randperm(len(tr_idx), device=device)
        ep_act = 0.0; ep_vf = 0.0; n_batches = 0
        model.policy.train()
        for s in range(0, len(tr_idx), args.batch_size):
            batch = tr_idx[rng_perm[s:s + args.batch_size].cpu().numpy()]
            ob = obs_t[batch]
            ac = act_t[batch]
            re = ret_t[batch]
            # Light Gaussian noise on inputs only (DART-style) for robustness
            # to mild state distribution shift at deploy time. Doesn't change
            # the regression target.
            if args.input_noise_std > 0:
                ob = ob + torch.randn_like(ob) * args.input_noise_std
            pred_a, pred_v = policy_forward(model.policy, ob)
            act_loss = F.mse_loss(pred_a, ac)
            vf_loss = F.mse_loss(pred_v, re)
            loss = act_loss + args.vf_coef * vf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_act += act_loss.item()
            ep_vf += vf_loss.item()
            n_batches += 1
        train_act = ep_act / max(n_batches, 1)
        train_vf = ep_vf / max(n_batches, 1)

        # Val (action MSE is what we early-stop on)
        model.policy.eval()
        with torch.no_grad():
            pred_a, pred_v = policy_forward(model.policy, obs_t[val_idx])
            val_act = F.mse_loss(pred_a, act_t[val_idx]).item()
            val_vf = F.mse_loss(pred_v, ret_t[val_idx]).item()
        is_best = val_act < best_val
        if is_best:
            best_val = val_act
            best_state = copy.deepcopy(model.policy.state_dict())
            best_epoch = epoch

        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d}  act={train_act:.6f}  vf={train_vf:.2f}"
                  f"  val_act={val_act:.6f}  val_vf={val_vf:.2f}"
                  + ("  ★" if is_best else ""))

    # Restore best-val weights
    model.policy.load_state_dict(best_state)
    print(f"  Restored best-val weights from epoch {best_epoch} (val={best_val:.6f})")

    # Shrink the policy's log_std so PPO's first stochastic rollouts stay
    # close to the BC mean. SB3 default log_std=0 → exploration stddev=1
    # which dwarfs the [-1,1] BC action range and instantly collapses the
    # policy. log_std=log(args.init_std) lets PPO entropy-anneal upward
    # if useful.
    with torch.no_grad():
        target = float(np.log(args.init_std))
        model.policy.log_std.fill_(target)
    print(f"  Set policy.log_std = log({args.init_std}) = {target:.3f}"
          f"  (stochastic action stddev ≈ {args.init_std} per dim)")

    # ── Save ───────────────────────────────────────────────────────────
    print("\n[4/4] Saving")
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "bc_model")
    vecnorm_path = os.path.join(args.save_dir, "vec_normalize.pkl")
    model.save(model_path)
    env.save(vecnorm_path)
    print(f"  Model:        {model_path}.zip")
    print(f"  VecNormalize: {vecnorm_path}")

    # ── Optional verification: run policy in env, count gates ──────────
    if args.verify:
        print("\n[5/4] Verifying BC policy in env (deterministic, 1 episode)")
        from stable_baselines3.common.vec_env import VecNormalize as VN
        # Reload to make sure save/load round-trips correctly
        eval_raw = FlightmareSB3VecEnv(flightgym.QuadrotorEnv_v1())
        eval_raw.set_random_reset(False)
        eval_env = VN.load(vecnorm_path, eval_raw)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_model = PPO.load(model_path)

        obs0 = eval_env.reset()
        ep_r = 0.0
        gates = 0
        for step in range(1500):
            action, _ = eval_model.predict(obs0, deterministic=True)
            obs0, r, d, info = eval_env.step(action)
            ep_r += r[0]
            gates = max(gates, int(info[0].get("gates_passed", 0)))
            if d[0]:
                print(f"  episode done at step {step}: gates={gates} R={ep_r:.1f}"
                      f" term={int(info[0].get('term_reason', -1))}")
                break
        else:
            print(f"  reached step cap: gates={gates} R={ep_r:.1f}")

    print("\nNext: fine-tune with PPO at low LR")
    print(f"  python gate_racing_iros.py --train 1 \\")
    print(f"    --load-model {model_path}.zip \\")
    print(f"    --load-vecnorm {vecnorm_path} \\")
    print(f"    --save-path saved/bc_then_ppo \\")
    print(f"    --tb-log saved/tb_logs/bc_then_ppo \\")
    print(f"    --total-timesteps 25000000")


if __name__ == "__main__":
    main()
