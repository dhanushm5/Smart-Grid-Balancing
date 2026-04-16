"""Tests for CityLearn helper utilities (no network access required)."""

import math
import tempfile
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import pytest

from smartgrid.citylearn_support import (
    CityLearnBackendConfig,
    _episode_result_from_kpis,
    _mean_episode_results,
    ensure_citylearn_schema,
)
from smartgrid.environment import EpisodeResult


class TestEnsureCityLearnSchema:
    def test_returns_explicit_path_when_file_exists(self, tmp_path: Path) -> None:
        schema = tmp_path / "schema.json"
        schema.write_text("{}")
        config = CityLearnBackendConfig(schema_path=schema)
        result = ensure_citylearn_schema(config)
        assert result == schema.resolve()

    def test_raises_when_explicit_path_missing(self, tmp_path: Path) -> None:
        config = CityLearnBackendConfig(schema_path=tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            ensure_citylearn_schema(config)


class TestEpisodeResultFromKpis:
    def _make_metrics(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_returns_defaults_for_empty_dataframe(self) -> None:
        empty = pd.DataFrame(columns=["cost_function", "name", "value"])
        result = _episode_result_from_kpis(empty)
        assert result.total_cost == 0.0
        assert result.total_emissions == 0.0

    def test_extracts_known_values(self) -> None:
        metrics = self._make_metrics([
            {"cost_function": "cost_total", "name": "District", "value": 42.5},
            {"cost_function": "carbon_emissions_total", "name": "District", "value": 10.0},
            {"cost_function": "all_time_peak_average", "name": "District", "value": 100.0},
        ])
        result = _episode_result_from_kpis(metrics)
        assert result.total_cost == 42.5
        assert result.total_emissions == 10.0
        assert result.peak_demand_kw == 100.0

    def test_handles_nan_values(self) -> None:
        metrics = self._make_metrics([
            {"cost_function": "cost_total", "name": "District", "value": float("nan")},
        ])
        result = _episode_result_from_kpis(metrics)
        assert result.total_cost == 0.0  # default


class TestMeanEpisodeResults:
    def test_averages_correctly(self) -> None:
        r1 = EpisodeResult(total_cost=10, total_emissions=20, peak_demand_kw=30,
                           mean_unserved_renewable_kw=5, comfort_violations=2, battery_throughput_kwh=40)
        r2 = EpisodeResult(total_cost=20, total_emissions=40, peak_demand_kw=50,
                           mean_unserved_renewable_kw=15, comfort_violations=4, battery_throughput_kwh=60)
        avg = _mean_episode_results([r1, r2])
        assert avg.total_cost == pytest.approx(15.0)
        assert avg.total_emissions == pytest.approx(30.0)
        assert avg.peak_demand_kw == pytest.approx(40.0)
        assert avg.battery_throughput_kwh == pytest.approx(50.0)
        assert avg.comfort_violations == 3  # int(round(mean(2, 4)))
