import math
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for a simplified grid + battery simulation."""

    battery_capacity_kwh: float = 150.0
    max_battery_power_kw: float = 40.0
    battery_round_trip_efficiency: float = 0.92
    initial_soc: float = 0.55
    timestep_hours: float = 1.0
    random_seed: int = 7

    # Observation normalisation denominators
    demand_norm_max: float = 150.0
    renewable_norm_max: float = 120.0

    # Comfort / safety thresholds
    soc_comfort_min: float = 0.10

    @property
    def charge_efficiency(self) -> float:
        """One-way efficiency applied when charging."""
        return math.sqrt(self.battery_round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        """One-way efficiency applied when discharging."""
        return math.sqrt(self.battery_round_trip_efficiency)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    horizon_days: int = 14
    # Profile generation
    demand_noise_std: float = 0.5
    renewable_share: float = 0.35
    peak_penalty_weight: float = 1.3
    energy_cost_weight: float = 1.0
    emissions_weight: float = 1.0
    battery_cycling_weight: float = 0.08
    num_eval_episodes: int = 5

    # Reward shaping
    peak_threshold_kw: float = 75.0
    comfort_penalty: float = 500.0
