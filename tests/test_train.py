"""Tests for the training loop (lightweight, 1-day episodes)."""

import numpy as np

from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import GridEnvironment
from smartgrid.policies import LinearPolicy
from smartgrid.train import train_linear_policy, TrainingHistory


class TestTrainLinearPolicy:
    def _make_env(self) -> GridEnvironment:
        return GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=1))

    def test_returns_policy_and_history(self) -> None:
        env = self._make_env()
        policy, history = train_linear_policy(env, episodes=5)
        assert isinstance(policy, LinearPolicy)
        assert isinstance(history, TrainingHistory)
        assert len(history.episode_rewards) == 5

    def test_reward_does_not_degrade_over_training(self) -> None:
        env = self._make_env()
        policy, history = train_linear_policy(env, episodes=30)
        # Allow modest noise: last quarter should not be drastically worse
        first_quarter = np.mean(history.episode_rewards[:8])
        last_quarter = np.mean(history.episode_rewards[-8:])
        # Tolerate up to 5% degradation (both values are negative costs)
        assert last_quarter >= first_quarter * 1.05

    def test_deterministic_across_runs(self) -> None:
        env1 = self._make_env()
        env2 = self._make_env()
        p1, h1 = train_linear_policy(env1, episodes=10)
        p2, h2 = train_linear_policy(env2, episodes=10)
        np.testing.assert_array_almost_equal(p1.weights, p2.weights)
        assert h1.episode_rewards == h2.episode_rewards

    def test_policy_weights_shape(self) -> None:
        env = self._make_env()
        policy, _ = train_linear_policy(env, episodes=3)
        assert policy.weights.shape == (6,)
