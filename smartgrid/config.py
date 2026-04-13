from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for a simplified grid + battery simulation."""

    battery_capacity_kwh: float = 120.0
    max_battery_power_kw: float = 40.0
    battery_round_trip_efficiency: float = 0.92
    initial_soc: float = 0.55
    timestep_hours: float = 1.0
    random_seed: int = 7


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    horizon_days: int = 14
    demand_noise_std: float = 4.5
    renewable_share: float = 0.35
    peak_penalty_weight: float = 1.3
    energy_cost_weight: float = 1.0
    emissions_weight: float = 1.0
    battery_cycling_weight: float = 0.08
    num_eval_episodes: int = 5
