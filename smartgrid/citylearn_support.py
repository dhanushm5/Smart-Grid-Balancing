from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import FileHandler

from .environment import EpisodeResult


CITYLEARN_REPOSITORY = "https://github.com/citylearn-project/CityLearn.git"
DEFAULT_CITYLEARN_DATASET = "citylearn_challenge_2022_phase_all_plus_evs"


@dataclass(frozen=True)
class CityLearnBackendConfig:
    dataset_name: str = DEFAULT_CITYLEARN_DATASET
    episode_time_steps: int = 96
    random_seed: int = 0
    cache_root: Path = Path.home() / ".cache" / "smartgrid_balancing" / "citylearn"
    schema_path: Path | None = None


def ensure_citylearn_schema(config: CityLearnBackendConfig) -> Path:
    """Return a local CityLearn schema path, cloning the upstream dataset if needed."""

    if config.schema_path is not None:
        schema_path = Path(config.schema_path).expanduser().resolve()
        if not schema_path.is_file():
            raise FileNotFoundError(f"CityLearn schema not found: {schema_path}")
        return schema_path

    dataset_dir = config.cache_root / config.dataset_name
    schema_path = dataset_dir / "schema.json"
    if schema_path.is_file():
        return schema_path

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="citylearn_clone_") as temp_dir:
        clone_dir = Path(temp_dir) / "CityLearn"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                CITYLEARN_REPOSITORY,
                str(clone_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set", f"data/datasets/{config.dataset_name}"],
            check=True,
        )

        source_dataset = clone_dir / "data" / "datasets" / config.dataset_name
        if not source_dataset.exists():
            raise FileNotFoundError(f"CityLearn dataset {config.dataset_name!r} was not found in the upstream repository clone.")

        shutil.copytree(source_dataset, dataset_dir, dirs_exist_ok=True)

    if not schema_path.is_file():
        raise FileNotFoundError(f"Failed to prepare CityLearn schema at {schema_path}")

    return schema_path


def build_citylearn_env(config: CityLearnBackendConfig) -> CityLearnEnv:
    schema_path = ensure_citylearn_schema(config)
    schema = FileHandler.read_json(schema_path)
    schema["root_directory"] = str(schema_path.parent)
    schema["central_agent"] = True
    return CityLearnEnv(schema, episode_time_steps=config.episode_time_steps, random_seed=config.random_seed)


class CityLearnHeuristicController:
    """Hand-tuned controller for CityLearn action vectors.

    Observation scales (from the 2022 challenge dataset):
    - ``hour``: 0-23 integer
    - ``electricity_pricing``: ~0.13-0.17  ($/kWh)
    - ``carbon_intensity``: ~0.16-0.25  (kgCO2/kWh)
    - ``solar_generation``: 0-10  (kWh per building per hour)
    - ``electrical_storage_soc``: 0.0-1.0
    - ``net_electricity_consumption``: can be negative when solar exceeds load

    Action convention (central agent, per-building slots):
    - ``electrical_storage``: positive = charge, negative = discharge, range [-1, 1]
    - ``electric_vehicle_storage_*``: positive = charge, range depends on charger
    """

    def __init__(self, env: CityLearnEnv) -> None:
        self.action_names = list(env.action_names[0])
        self.observation_names = list(env.observation_names[0])
        self.action_low = np.array(env.action_space[0].low, dtype=float)
        self.action_high = np.array(env.action_space[0].high, dtype=float)
        self._first_indices: dict[str, int] = {}
        for index, name in enumerate(self.observation_names):
            self._first_indices.setdefault(name, index)

    def act(self, observation: Iterable[float]) -> list[list[float]]:
        observation_values = np.asarray(list(observation), dtype=float)

        hour = self._first_value(observation_values, "hour", default=12.0)
        price = self._first_value(observation_values, "electricity_pricing", default=0.15)
        carbon = self._first_value(observation_values, "carbon_intensity", default=0.20)
        solar = self._first_value(observation_values, "solar_generation", default=0.0)
        net_consumption = self._first_value(observation_values, "net_electricity_consumption", default=0.0)
        storage_soc = self._mean_value(observation_values, "electrical_storage_soc", default=0.0)

        # --- Battery storage strategy ---
        storage_signal = 0.0

        # Evening peak (17-21h): discharge if we have stored energy
        if 17 <= hour <= 21 and storage_soc > 0.15:
            # Discharge harder when SOC is high and price/carbon are elevated
            if storage_soc > 0.5:
                storage_signal = -0.8
            else:
                storage_signal = -0.45

        # Midday solar surplus (9-15h): charge aggressively from cheap solar
        elif 9 <= hour <= 15 and storage_soc < 0.90:
            if solar > 2.0:
                # Strong solar — charge aggressively
                storage_signal = 0.7
            elif solar > 0.5 and price < 0.15:
                # Moderate solar + cheap price — charge moderately
                storage_signal = 0.4
            elif price < 0.14:
                # No solar but cheap electricity — mild charge
                storage_signal = 0.25

        # Early morning (0-6h): mild charge during off-peak if battery is low
        elif 0 <= hour <= 6 and storage_soc < 0.4 and price < 0.16:
            storage_signal = 0.3

        # Late afternoon (15-17h): hold / top up if solar still available
        elif 15 <= hour < 17 and storage_soc < 0.7 and solar > 1.0:
            storage_signal = 0.35

        # --- EV charging strategy ---
        # Charge EVs if they're connected and need charge before departure
        ev_signal = self._compute_ev_signal(observation_values, hour)

        # --- Build action vector ---
        actions = np.zeros(len(self.action_names), dtype=float)
        for index, action_name in enumerate(self.action_names):
            if action_name == "electrical_storage":
                actions[index] = self._scale_action(index, storage_signal)
            elif action_name.startswith("electric_vehicle_storage_"):
                actions[index] = self._scale_action(index, ev_signal)
            else:
                actions[index] = 0.0

        return [actions.tolist()]

    def _compute_ev_signal(self, observation: np.ndarray, hour: float) -> float:
        """Simple EV charging: charge during off-peak / solar hours."""
        # Charge EVs during solar hours or overnight off-peak
        if 9 <= hour <= 15 or 0 <= hour <= 5:
            return 0.3
        return 0.0

    def _first_value(self, observation: np.ndarray, name: str, default: float = 0.0) -> float:
        index = self._first_indices.get(name)
        if index is None or index >= len(observation):
            return default
        return float(observation[index])

    def _mean_value(self, observation: np.ndarray, name: str, default: float = 0.0) -> float:
        values = [float(observation[index]) for index, feature_name in enumerate(self.observation_names) if feature_name == name and index < len(observation)]
        if not values:
            return default
        return float(mean(values))

    def _scale_action(self, index: int, normalized_value: float) -> float:
        low = float(self.action_low[index])
        high = float(self.action_high[index])

        clipped = float(np.clip(normalized_value, -1.0, 1.0))
        if low >= 0.0:
            # Action space is [0, high] — map positive half only
            return float(max(0.0, clipped) * high)

        if clipped >= 0.0:
            return clipped * high

        return clipped * abs(low)


def evaluate_citylearn_controller(env_config: CityLearnBackendConfig, episodes: int = 3) -> tuple[EpisodeResult, EpisodeResult]:
    baseline_results: list[EpisodeResult] = []
    candidate_results: list[EpisodeResult] = []

    for offset in range(episodes):
        baseline_results.append(_run_citylearn_episode(env_config, policy_name="baseline", seed=env_config.random_seed + offset))
        candidate_results.append(_run_citylearn_episode(env_config, policy_name="heuristic", seed=env_config.random_seed + offset))

    return _mean_episode_results(baseline_results), _mean_episode_results(candidate_results)


def _run_citylearn_episode(env_config: CityLearnBackendConfig, policy_name: str, seed: int) -> EpisodeResult:
    env = build_citylearn_env(CityLearnBackendConfig(
        dataset_name=env_config.dataset_name,
        episode_time_steps=env_config.episode_time_steps,
        random_seed=seed,
        cache_root=env_config.cache_root,
        schema_path=env_config.schema_path,
    ))

    try:
        observations, _ = env.reset(seed=seed)
        controller = CityLearnHeuristicController(env)

        while not env.terminated:
            if policy_name == "baseline":
                actions = [np.zeros(env.action_space[0].shape[0], dtype=float).tolist()]
            else:
                actions = controller.act(observations[0])

            observations, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        metrics = env.evaluate()
        return _episode_result_from_kpis(metrics)
    finally:
        env.close()


def _episode_result_from_kpis(metrics) -> EpisodeResult:
    def _value(name: str, default: float = 0.0) -> float:
        rows = metrics[(metrics["cost_function"] == name) & (metrics["name"] == "District")]
        if rows.empty:
            return default
        value = float(rows["value"].iloc[0])
        if math.isnan(value):
            return default
        return value

    return EpisodeResult(
        total_cost=_value("cost_total"),
        total_emissions=_value("carbon_emissions_total"),
        peak_demand_kw=_value("all_time_peak_average"),
        mean_unserved_renewable_kw=_value("annual_normalized_unserved_energy_total"),
        comfort_violations=int(round(_value("discomfort_proportion", default=0.0))),
        battery_throughput_kwh=_value("electricity_consumption_total"),
    )


def _mean_episode_results(results: list[EpisodeResult]) -> EpisodeResult:
    return EpisodeResult(
        total_cost=mean(result.total_cost for result in results),
        total_emissions=mean(result.total_emissions for result in results),
        peak_demand_kw=mean(result.peak_demand_kw for result in results),
        mean_unserved_renewable_kw=mean(result.mean_unserved_renewable_kw for result in results),
        comfort_violations=int(round(mean(result.comfort_violations for result in results))),
        battery_throughput_kwh=mean(result.battery_throughput_kwh for result in results),
    )