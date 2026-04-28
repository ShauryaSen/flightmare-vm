"""Stable-Baselines3 VecEnv wrapper with observation ablation support.

Each ablation experiment is defined by an AblationConfig that specifies:
  - which raw obs indices to keep (true dimensionality reduction, not masking)
  - an optional noise function applied after slicing (for Exp G)
  - an optional transform applied after slicing (for Exp H: euler → rot matrix)

Full raw observation layout (15-dim from C++ env):
  [0:3]  position (x, y, z)  — world frame
  [3:6]  euler angles (roll, pitch, yaw) — world frame
  [6:9]  linear velocity (vx, vy, vz)   — body frame
  [9:12] angular velocity (wx, wy, wz)  — body frame (body rates)
  [12:15] relative gate position (dx, dy, dz) — body frame
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


# ── Ablation definitions ─────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Defines a single observation ablation condition.

    Attributes:
        name: Short identifier used in directory names and logs.
        description: Human-readable description for logging.
        obs_indices: Indices into the raw 15-dim obs to keep.
            Use None to keep all indices (full baseline).
        noise_fn: Optional callable (obs_array) -> obs_array applied after
            slicing. Receives shape (n_envs, len(obs_indices)), returns same.
        transform_fn: Optional callable (obs_array) -> obs_array applied after
            slicing and noise. Can change output dimensionality (e.g., Exp H:
            euler(3) → rot_matrix_flat(9) adds 6 dims). If provided, the
            caller is responsible for computing the resulting obs_dim.
        obs_dim_override: If transform_fn changes the dimension, set this to
            the final observation dimension. If None, inferred from obs_indices.
    """
    name: str
    description: str
    obs_indices: Optional[List[int]]       # None = all 15 dims
    noise_fn: Optional[Callable] = None    # applied after slicing
    transform_fn: Optional[Callable] = None  # can change dim (Exp H)
    obs_dim_override: Optional[int] = None

    @property
    def obs_dim(self) -> int:
        if self.obs_dim_override is not None:
            return self.obs_dim_override
        if self.obs_indices is None:
            return 15
        return len(self.obs_indices)


# ── Experiment registry ───────────────────────────────────────────────────────

def _gate_noise(sigma: float) -> Callable:
    """Returns a noise function that adds Gaussian noise to gate dims [12:15].

    Note: this function receives the *already-sliced* obs, so gate dims are
    always the LAST 3 elements (since all Exp G conditions keep gate obs).
    """
    def _fn(obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        out[:, -3:] += np.random.normal(0.0, sigma, (obs.shape[0], 3)).astype(np.float32)
        return out
    return _fn


def _euler_to_rotmat_transform(obs: np.ndarray) -> np.ndarray:
    """Replace euler(3) in obs positions [3:6] with flat rotation matrix (9).

    Input obs layout (18 raw dims with euler): pos(3)|euler(3)|linv(3)|angv(3)|gate(3)
    We receive: pos(3)|euler(3)|linv(3)|angv(3)|gate(3)  (15 dims, same as baseline)
    Output:     pos(3)|R_flat(9)|linv(3)|angv(3)|gate(3) (21 dims)
    """
    n = obs.shape[0]
    pos = obs[:, 0:3]
    euler = obs[:, 3:6]  # roll, pitch, yaw
    linv = obs[:, 6:9]
    angv = obs[:, 9:12]
    gate = obs[:, 12:15]

    # Build rotation matrices from euler angles (ZYX convention)
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.stack([
        cy*cp,          cy*sp*sr - sy*cr,   cy*sp*cr + sy*sr,
        sy*cp,          sy*sp*sr + cy*cr,   sy*sp*cr - cy*sr,
        -sp,            cp*sr,              cp*cr,
    ], axis=1)  # (n, 9)

    return np.concatenate([pos, R, linv, angv, gate], axis=1).astype(np.float32)


# Full registry of all ablation experiments
ABLATIONS: dict[str, AblationConfig] = {

    # ── A: Full baseline ──────────────────────────────────────────────────
    "A_full": AblationConfig(
        name="A_full",
        description="Full baseline: pos(3)|euler(3)|linv(3)|angv(3)|gate(3) — 15 dims",
        obs_indices=None,  # all 15
    ),

    # ── B: No euler angles ────────────────────────────────────────────────
    "B_no_euler": AblationConfig(
        name="B_no_euler",
        description="No orientation: pos(3)|linv(3)|angv(3)|gate(3) — 12 dims",
        obs_indices=list(range(0, 3)) + list(range(6, 15)),  # skip [3:6]
    ),

    # ── C: No angular velocity ────────────────────────────────────────────
    "C_no_angvel": AblationConfig(
        name="C_no_angvel",
        description="No gyro: pos(3)|euler(3)|linv(3)|gate(3) — 12 dims",
        obs_indices=list(range(0, 9)) + list(range(12, 15)),  # skip [9:12]
    ),

    # ── D: No gate observation (blind racing) ─────────────────────────────
    "D_no_gate": AblationConfig(
        name="D_no_gate",
        description="No gate obs: pos(3)|euler(3)|linv(3)|angv(3) — 12 dims",
        obs_indices=list(range(0, 12)),  # skip [12:15]
    ),

    # ── E: Gate + linear velocity only (minimal reactive) ─────────────────
    "E_gate_vel_only": AblationConfig(
        name="E_gate_vel_only",
        description="Minimal reactive: linv(3)|gate(3) — 6 dims",
        obs_indices=list(range(6, 9)) + list(range(12, 15)),
    ),

    # ── F: No absolute position (egocentric only) ─────────────────────────
    "F_no_pos": AblationConfig(
        name="F_no_pos",
        description="No position: euler(3)|linv(3)|angv(3)|gate(3) — 12 dims",
        obs_indices=list(range(3, 15)),  # skip [0:3]
    ),

    # ── G: Noisy gate observation ─────────────────────────────────────────
    "G1_noise_01": AblationConfig(
        name="G1_noise_01",
        description="Noisy gate σ=0.10m: full obs + N(0, 0.1²) on gate — 15 dims",
        obs_indices=None,
        noise_fn=_gate_noise(0.10),
    ),
    "G2_noise_03": AblationConfig(
        name="G2_noise_03",
        description="Noisy gate σ=0.30m: full obs + N(0, 0.3²) on gate — 15 dims",
        obs_indices=None,
        noise_fn=_gate_noise(0.30),
    ),
    "G3_noise_05": AblationConfig(
        name="G3_noise_05",
        description="Noisy gate σ=0.50m: full obs + N(0, 0.5²) on gate — 15 dims",
        obs_indices=None,
        noise_fn=_gate_noise(0.50),
    ),
    "G4_noise_10": AblationConfig(
        name="G4_noise_10",
        description="Noisy gate σ=1.00m: full obs + N(0, 1.0²) on gate — 15 dims",
        obs_indices=None,
        noise_fn=_gate_noise(1.00),
    ),

    # ── H: Rotation matrix instead of euler ───────────────────────────────
    "H_rotmat": AblationConfig(
        name="H_rotmat",
        description="Rotation matrix: pos(3)|R_flat(9)|linv(3)|angv(3)|gate(3) — 21 dims",
        obs_indices=None,  # pass all 15 to transform_fn
        transform_fn=_euler_to_rotmat_transform,
        obs_dim_override=21,
    ),

    # ── I: IMU-realistic (linv + angv + gate) ─────────────────────────────
    "I_imu_realistic": AblationConfig(
        name="I_imu_realistic",
        description="IMU-realistic: linv(3)|angv(3)|gate(3) — 9 dims",
        obs_indices=list(range(6, 15)),  # linv + angv + gate
    ),
}


# ── VecEnv wrapper ────────────────────────────────────────────────────────────

class AblationVecEnv(VecEnv):
    """VecEnv wrapper with observation ablation.

    Wraps the Flightmare C++ VecEnv and applies the specified AblationConfig
    to the raw 15-dim observation at every step and reset.

    Observation pipeline per step:
      1. C++ step() advances physics, auto-resets done envs
      2. getObs() reads actual updated state (pybind11 Ref workaround)
      3. Slice raw obs to keep only obs_indices (true dim reduction)
      4. Apply noise_fn if configured (Exp G)
      5. Apply transform_fn if configured (Exp H)
    """

    def __init__(self, env_wrapper, ablation: AblationConfig):
        self.wrapper = env_wrapper
        self.ablation = ablation

        n_envs = self.wrapper.getNumOfEnvs()
        raw_obs_dim = self.wrapper.getObsDim()  # always 15
        act_dim = self.wrapper.getActDim()

        assert raw_obs_dim == 15, (
            f"Expected 15-dim raw obs from C++ env, got {raw_obs_dim}. "
            "Check that you're using the IROS 2021 quadrotor_env build."
        )

        self._raw_obs_dim = raw_obs_dim
        self._ablation_obs_dim = ablation.obs_dim

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._ablation_obs_dim,),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        super().__init__(n_envs, observation_space, action_space)

        # Raw obs buffer — always 15-dim regardless of ablation
        self._raw_obs = np.zeros((n_envs, raw_obs_dim), dtype=np.float32)
        self._reward = np.zeros(n_envs, dtype=np.float32)
        self._done = np.zeros(n_envs, dtype=bool)

        extra_info_names = self.wrapper.getExtraInfoNames()
        self._extra_info = np.zeros(
            (n_envs, len(extra_info_names)), dtype=np.float32
        )
        self._extra_info_names = extra_info_names

        # Precompute index array for fast slicing (or None for all-15)
        if ablation.obs_indices is not None:
            self._idx = np.array(ablation.obs_indices, dtype=np.intp)
        else:
            self._idx = None

        print(f"[AblationVecEnv] Experiment: {ablation.name}")
        print(f"  {ablation.description}")
        print(f"  Raw obs dim:      {raw_obs_dim}")
        print(f"  Ablated obs dim:  {self._ablation_obs_dim}")
        print(f"  Num envs:         {n_envs}")
        print(f"  Noise fn:         {'YES' if ablation.noise_fn else 'no'}")
        print(f"  Transform fn:     {'YES' if ablation.transform_fn else 'no'}")

    def _apply_ablation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Apply index slicing → noise → transform to raw obs."""
        # 1. Slice (or keep all)
        if self._idx is not None:
            obs = raw_obs[:, self._idx]
        else:
            obs = raw_obs  # view, not copy — noise_fn will copy if needed

        # 2. Noise
        if self.ablation.noise_fn is not None:
            obs = self.ablation.noise_fn(obs)
        else:
            obs = obs.copy()  # ensure we return an independent array

        # 3. Transform (e.g., euler → rotation matrix)
        if self.ablation.transform_fn is not None:
            # transform_fn receives the full raw obs so it can access all dims
            # Note: for Exp H we pass the full raw_obs, not the sliced one
            obs = self.ablation.transform_fn(raw_obs)

        return obs.astype(np.float32)

    def step_async(self, actions):
        # Clip actions to the action space bounds to prevent physics explosions!
        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        self._actions = clipped_actions.astype(np.float32)

    def step_wait(self):
        # C++ step() advances physics and auto-resets done envs.
        # The obs output from step() is unreliable due to pybind11 Ref<>
        # temporary issue — we call getObs() explicitly to get the real state.
        self.wrapper.step(
            self._actions, self._raw_obs,
            self._reward, self._done, self._extra_info
        )
        self.wrapper.getObs(self._raw_obs)

        # ── NaN sanitization ─────────────────────────────────────────────
        # Rarely, the C++ physics produces NaN states (degenerate quaternion,
        # extreme velocity). If any env's obs contains NaN/Inf, zero it out,
        # zero the reward, and force done=True so the env auto-resets next step.
        nan_mask = ~np.isfinite(self._raw_obs).all(axis=1)  # (n_envs,)
        if nan_mask.any():
            n_bad = int(nan_mask.sum())
            if n_bad > 0:
                self._raw_obs[nan_mask] = 0.0
                self._reward[nan_mask] = 0.0
                self._done[nan_mask] = True

        obs_out = self._apply_ablation(self._raw_obs)

        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            for j, name in enumerate(self._extra_info_names):
                infos[i][name] = self._extra_info[i, j]

        return obs_out, self._reward.copy(), self._done.copy(), infos

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._raw_obs)
        # Sanitize reset obs too — degenerate initial states are possible
        nan_mask = ~np.isfinite(self._raw_obs).all(axis=1)
        if nan_mask.any():
            self._raw_obs[nan_mask] = 0.0
        return self._apply_ablation(self._raw_obs)

    def close(self):
        self.wrapper.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.wrapper.setSeed(seed)
