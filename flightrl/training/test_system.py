#!/usr/bin/env python3
"""Integration tests for the drone racing RL pipeline (Tests 1a, 7).

Tests the full Python → pybind → C++ → Python data path.

Usage: pytest test_system.py -v
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import flightgym
from sb3_vec_env import FlightmareSB3VecEnv


@pytest.fixture
def env():
    cpp = flightgym.QuadrotorEnv_v1()
    wrapper = FlightmareSB3VecEnv(cpp)
    yield wrapper, cpp
    wrapper.close()


class TestObsFreshness:
    """Test 7: Verify getObs() workaround produces fresh observations."""

    def test_obs_matches_getobs(self, env):
        wrapper, cpp = env
        obs = wrapper.reset()
        for _ in range(50):
            action = np.zeros((wrapper.num_envs, 4), dtype=np.float32)
            obs_step, rew, done, info = wrapper.step(action)
            obs_verify = np.zeros_like(obs_step)
            cpp.getObs(obs_verify)
            np.testing.assert_allclose(obs_step, obs_verify, atol=1e-5,
                err_msg="step() obs differs from getObs() — Bug 2 workaround broken")

    def test_obs_changes_between_steps(self, env):
        wrapper, _ = env
        obs_prev = wrapper.reset().copy()
        action = np.zeros((wrapper.num_envs, 4), dtype=np.float32)
        obs_next, _, _, _ = wrapper.step(action)
        # At least acceleration obs should change (gravity acts)
        assert not np.allclose(obs_prev[0, 3:6], obs_next[0, 3:6], atol=1e-6), \
            "Observations didn't change after a step — physics may not be advancing"


class TestAutoReset:
    """Test 7b: Verify auto-reset produces valid observations."""

    def test_autoreset_valid_obs(self, env):
        wrapper, _ = env
        obs = wrapper.reset()
        for step in range(5000):
            action = np.random.uniform(-1, 1, (wrapper.num_envs, 4)).astype(np.float32)
            obs, rew, done, info = wrapper.step(action)
            if done.any():
                for i in np.where(done)[0]:
                    assert np.all(np.isfinite(obs[i])), \
                        f"Env {i} has NaN obs after auto-reset"
                    assert obs[i, 18] > 0.1, \
                        f"Gate distance {obs[i,18]} too small after reset"
                break
        else:
            pytest.skip("No env terminated in 5000 steps")


class TestObsDimensions:
    """Test 5a: Verify obs/act dimensions match expected."""

    def test_dimensions(self, env):
        wrapper, _ = env
        assert wrapper.observation_space.shape == (26,)
        assert wrapper.action_space.shape == (4,)

    def test_obs_finite_after_reset(self, env):
        wrapper, _ = env
        obs = wrapper.reset()
        assert np.all(np.isfinite(obs)), "Obs contains NaN/inf after reset"


class TestRewardPipeline:
    """Test 1a: Verify reward reaches Python correctly."""

    def test_hover_reward_near_zero(self, env):
        """Hover (zero action) should produce small reward (just survival)."""
        wrapper, _ = env
        wrapper.reset()
        total = np.zeros(wrapper.num_envs)
        for _ in range(10):
            action = np.zeros((wrapper.num_envs, 4), dtype=np.float32)
            _, rew, done, _ = wrapper.step(action)
            total += rew
        # With survival_bonus=0.1, 10 steps → ~1.0 per env
        # Some envs may crash, so just check mean is reasonable
        alive = total[total > -10]
        if len(alive) > 0:
            assert np.mean(alive) > -1.0, \
                f"Mean hover reward {np.mean(alive)} is too negative"

    def test_reward_not_always_zero(self, env):
        wrapper, _ = env
        wrapper.reset()
        all_rew = []
        for _ in range(100):
            action = np.random.uniform(-0.5, 0.5, (wrapper.num_envs, 4)).astype(np.float32)
            _, rew, _, _ = wrapper.step(action)
            all_rew.append(rew)
        all_rew = np.array(all_rew)
        assert np.any(np.abs(all_rew) > 1e-6), \
            "All rewards are zero — reward function may be broken"


class TestExtraInfo:
    """Verify extra_info fields are populated through pybind."""

    def test_extra_info_available(self, env):
        wrapper, _ = env
        assert len(wrapper._extra_info_names) > 0, \
            "No extra_info names — updateExtraInfo() not working"
        assert "r_total" in wrapper._extra_info_names
        assert "r_progress" in wrapper._extra_info_names

    def test_extra_info_populated(self, env):
        wrapper, _ = env
        wrapper.reset()
        action = np.zeros((wrapper.num_envs, 4), dtype=np.float32)
        _, _, _, infos = wrapper.step(action)
        assert "r_total" in infos[0]
        assert "gate_idx" in infos[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
