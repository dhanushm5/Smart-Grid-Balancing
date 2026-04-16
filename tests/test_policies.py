"""Tests for ZeroActionPolicy, RuleBasedPeakShavingPolicy, and LinearPolicy."""

import numpy as np
import pytest

from smartgrid.policies import (
    LinearPolicy,
    RuleBasedPeakShavingPolicy,
    ZeroActionPolicy,
)


class TestZeroActionPolicy:
    def test_always_returns_zero(self) -> None:
        policy = ZeroActionPolicy()
        rng = np.random.default_rng(99)
        for _ in range(20):
            obs = rng.uniform(-2, 2, size=6)
            assert policy.act(obs) == 0.0


class TestRuleBasedPeakShavingPolicy:
    def test_output_is_bounded(self) -> None:
        policy = RuleBasedPeakShavingPolicy()
        rng = np.random.default_rng(7)
        for _ in range(50):
            obs = rng.uniform(0, 1, size=6)
            action = policy.act(obs)
            assert -1.0 <= action <= 1.0

    def test_discharges_on_high_demand_high_soc(self) -> None:
        # demand_norm > 0.72, soc > 0.12 -> should discharge (positive action)
        obs = np.array([0.80, 0.10, 0.15, 0.25, 0.30, 0.50])
        policy = RuleBasedPeakShavingPolicy()
        assert policy.act(obs) > 0.0

    def test_charges_on_high_renewables_low_demand(self) -> None:
        # demand_norm < 0.52, renewable_norm > 0.45, soc < 0.92
        obs = np.array([0.40, 0.60, 0.10, 0.15, 0.30, 0.50])
        policy = RuleBasedPeakShavingPolicy()
        assert policy.act(obs) < 0.0

    def test_charges_on_low_emissions_low_price(self) -> None:
        # demand_norm < 0.52, emissions < 0.20, price < 0.14, soc < 0.92
        obs = np.array([0.40, 0.20, 0.10, 0.15, 0.30, 0.50])
        policy = RuleBasedPeakShavingPolicy()
        assert policy.act(obs) < 0.0

    def test_idle_on_moderate_conditions(self) -> None:
        # demand_norm between 0.52 and 0.72, no trigger met
        obs = np.array([0.60, 0.30, 0.20, 0.30, 0.30, 0.50])
        policy = RuleBasedPeakShavingPolicy()
        assert policy.act(obs) == 0.0


class TestLinearPolicy:
    def test_output_bounded_by_tanh(self) -> None:
        rng = np.random.default_rng(12)
        weights = rng.normal(0, 1, size=6)
        policy = LinearPolicy(weights=weights, bias=0.5)
        for _ in range(50):
            obs = rng.uniform(-5, 5, size=6)
            action = policy.act(obs)
            assert -1.0 <= action <= 1.0

    def test_deterministic(self) -> None:
        weights = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.15])
        policy = LinearPolicy(weights=weights, bias=0.0)
        obs = np.array([1.0, 0.5, 0.2, 0.3, 0.8, 0.6])
        a1 = policy.act(obs)
        a2 = policy.act(obs)
        assert a1 == a2

    def test_zero_weights_returns_tanh_bias(self) -> None:
        policy = LinearPolicy(weights=np.zeros(6), bias=1.0)
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        assert policy.act(obs) == pytest.approx(float(np.tanh(1.0)), rel=1e-6)
