"""Shared pytest fixtures for the smart-grid test suite."""

import pytest

from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import GridEnvironment


@pytest.fixture
def grid_config() -> GridConfig:
    return GridConfig()


@pytest.fixture
def exp_config() -> ExperimentConfig:
    return ExperimentConfig(horizon_days=2)


@pytest.fixture
def env(grid_config: GridConfig, exp_config: ExperimentConfig) -> GridEnvironment:
    """A reset environment with a 2-day horizon (fast tests)."""
    environment = GridEnvironment(grid_config, exp_config)
    environment.reset(seed=42)
    return environment


@pytest.fixture
def short_env() -> GridEnvironment:
    """A 1-day environment suitable for quick training tests."""
    environment = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=1))
    environment.reset(seed=0)
    return environment
