"""Tests for GridConfig and ExperimentConfig dataclasses."""

import math

from smartgrid.config import ExperimentConfig, GridConfig


class TestGridConfigDefaults:
    def test_battery_capacity(self) -> None:
        cfg = GridConfig()
        assert cfg.battery_capacity_kwh == 120.0

    def test_initial_soc_fraction(self) -> None:
        cfg = GridConfig()
        assert 0.0 <= cfg.initial_soc <= 1.0

    def test_round_trip_efficiency_within_bounds(self) -> None:
        cfg = GridConfig()
        assert 0.0 < cfg.battery_round_trip_efficiency <= 1.0

    def test_charge_discharge_efficiency_symmetric(self) -> None:
        cfg = GridConfig()
        assert cfg.charge_efficiency == cfg.discharge_efficiency
        assert math.isclose(
            cfg.charge_efficiency * cfg.discharge_efficiency,
            cfg.battery_round_trip_efficiency,
            rel_tol=1e-9,
        )


class TestGridConfigCustom:
    def test_custom_capacity(self) -> None:
        cfg = GridConfig(battery_capacity_kwh=200.0)
        assert cfg.battery_capacity_kwh == 200.0

    def test_custom_efficiency(self) -> None:
        cfg = GridConfig(battery_round_trip_efficiency=0.85)
        expected_one_way = math.sqrt(0.85)
        assert math.isclose(cfg.charge_efficiency, expected_one_way, rel_tol=1e-9)

    def test_custom_soc_comfort_threshold(self) -> None:
        cfg = GridConfig(soc_comfort_min=0.20)
        assert cfg.soc_comfort_min == 0.20


class TestExperimentConfigDefaults:
    def test_horizon_days(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.horizon_days == 14

    def test_peak_threshold(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.peak_threshold_kw == 95.0

    def test_comfort_penalty(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.comfort_penalty == 5.0

    def test_weights_are_positive(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.energy_cost_weight > 0
        assert cfg.emissions_weight > 0
        assert cfg.peak_penalty_weight > 0
        assert cfg.battery_cycling_weight >= 0
