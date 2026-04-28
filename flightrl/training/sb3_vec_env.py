"""Stable-Baselines3 compatible wrapper for Flightmare's C++ VecEnv."""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class FlightmareSB3VecEnv(VecEnv):
    """Wraps flightgym's C++ vectorized env for SB3 compatibility.

    Note: The C++ VecEnv::step() calls perAgentStep() which internally calls
    QuadrotorEnv::step(act.row(i), obs.row(i)). Due to an Eigen/pybind11
    issue, the obs.row(i) creates a temporary when mapped to Ref<Vector<>>,
    so observation updates from step() are lost. We work around this by
    calling getObs() explicitly after step() to read the actual internal state.
    """

    def __init__(self, env_wrapper):
        self.wrapper = env_wrapper
        n_envs = self.wrapper.getNumOfEnvs()
        obs_dim = self.wrapper.getObsDim()
        act_dim = self.wrapper.getActDim()

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        super().__init__(n_envs, observation_space, action_space)

        self._observation = np.zeros((n_envs, obs_dim), dtype=np.float32)
        self._reward = np.zeros(n_envs, dtype=np.float32)
        self._done = np.zeros(n_envs, dtype=bool)

        extra_info_names = self.wrapper.getExtraInfoNames()
        self._extra_info = np.zeros((n_envs, len(extra_info_names)), dtype=np.float32)
        self._extra_info_names = extra_info_names

    def step_async(self, actions):
        self._actions = actions.astype(np.float32)

    def step_wait(self):
        # C++ step() advances physics, computes reward/done, and auto-resets
        # done envs. However, due to Eigen Ref<Vector<>> vs RowMajor::row()
        # mismatch, the obs output from step() is written to a temporary.
        # We must call getObs() separately to read the actual updated state.
        self.wrapper.step(
            self._actions, self._observation, self._reward, self._done, self._extra_info
        )

        # Fetch actual observations from the internal env state
        self.wrapper.getObs(self._observation)

        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            for j, name in enumerate(self._extra_info_names):
                infos[i][name] = self._extra_info[i, j]

        return self._observation.copy(), self._reward.copy(), self._done.copy(), infos

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()

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
