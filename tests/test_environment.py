"""Expanded environment tests — edge cases, invariants, and regression checks."""

import numpy as np

from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import GridEnvironment
from smartgrid.policies import RuleBasedPeakShavingPolicy, ZeroActionPolicy


def run_episode(env: GridEnvironment, policy) -> float:
    obs, _ = env.reset(seed=123)
    done = False
    total_reward = 0.0

    while not done:
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        done = terminated or truncated
        total_reward += reward

    return total_reward


# ── Original tests (preserved) ───────────────────────────────────────────────


def test_environment_episode_runs_to_completion() -> None:
    env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=2))
    reward = run_episode(env, ZeroActionPolicy())
    summary = env.summarize_episode()

    assert isinstance(reward, float)
    assert summary.peak_demand_kw > 0.0


def test_rule_based_policy_improves_peak() -> None:
    env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=3))

    run_episode(env, ZeroActionPolicy())
    baseline = env.summarize_episode()

    run_episode(env, RuleBasedPeakShavingPolicy())
    candidate = env.summarize_episode()

    assert candidate.peak_demand_kw <= baseline.peak_demand_kw


# ── New tests ─────────────────────────────────────────────────────────────────


class TestObservation:
    def test_reset_returns_correct_shape(self, env: GridEnvironment) -> None:
        obs, _ = env.reset(seed=99)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_shape(self, env: GridEnvironment) -> None:
        obs, reward, terminated, truncated, info = env.step(0.0)
        done = terminated or truncated
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestBatterySOC:
    def test_soc_stays_within_bounds_on_max_discharge(self) -> None:
        cfg = GridConfig(initial_soc=0.10)  # start almost empty
        env = GridEnvironment(cfg, ExperimentConfig(horizon_days=1))
        env.reset(seed=0)
        for _ in range(24):
            env.step(1.0)  # max discharge every step
        soc = env._soc_kwh / cfg.battery_capacity_kwh
        assert soc >= 0.0

    def test_soc_stays_within_bounds_on_max_charge(self) -> None:
        cfg = GridConfig(initial_soc=0.95)  # start almost full
        env = GridEnvironment(cfg, ExperimentConfig(horizon_days=1))
        env.reset(seed=0)
        for _ in range(24):
            env.step(-1.0)  # max charge every step
        soc = env._soc_kwh / cfg.battery_capacity_kwh
        assert soc <= 1.0


class TestEpisodeTermination:
    def test_terminates_at_expected_step(self) -> None:
        days = 3
        env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=days))
        env.reset(seed=0)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0.0)
            done = terminated or truncated
            steps += 1
        assert steps == 24 * days


class TestActionClipping:
    def test_extreme_actions_are_clipped(self, env: GridEnvironment) -> None:
        # Actions far outside [-1, 1] should not crash
        obs1, r1, _, _, _ = env.step(5.0)
        assert obs1.shape == (6,)

        env.reset(seed=42)
        obs2, r2, _, _, _ = env.step(-10.0)
        assert obs2.shape == (6,)


class TestReward:
    def test_reward_is_always_nonpositive(self, env: GridEnvironment) -> None:
        """All reward components are costs, so reward <= 0."""
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(0.0)
            done = terminated or truncated
            assert reward <= 0.0, f"Reward should be <= 0, got {reward}"


class TestEpisodeSummary:
    def test_values_are_finite(self, env: GridEnvironment) -> None:
        run_episode(env, ZeroActionPolicy())
        summary = env.summarize_episode()
        assert np.isfinite(summary.total_cost)
        assert np.isfinite(summary.total_emissions)
        assert np.isfinite(summary.peak_demand_kw)
        assert np.isfinite(summary.mean_unserved_renewable_kw)
        assert np.isfinite(summary.battery_throughput_kwh)

    def test_battery_throughput_zero_for_no_control(self, env: GridEnvironment) -> None:
        run_episode(env, ZeroActionPolicy())
        summary = env.summarize_episode()
        assert summary.battery_throughput_kwh == 0.0


class TestDeterminism:
    def test_same_seed_same_profiles(self) -> None:
        env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=2))
        obs1, _ = env.reset(seed=77)
        obs2, _ = env.reset(seed=77)
        np.testing.assert_array_equal(obs1, obs2)
