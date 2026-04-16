from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import ExperimentConfig, GridConfig


@dataclass
class EpisodeResult:
    total_cost: float
    total_emissions: float
    peak_demand_kw: float
    mean_unserved_renewable_kw: float
    comfort_violations: int
    battery_throughput_kwh: float


class GridEnvironment:
    """A lightweight, RL-friendly smart-grid environment with battery dispatch."""

    def __init__(self, grid_cfg: GridConfig, exp_cfg: ExperimentConfig) -> None:
        self.grid_cfg = grid_cfg
        self.exp_cfg = exp_cfg
        self.steps_per_day = int(24 / grid_cfg.timestep_hours)
        self.episode_steps = self.steps_per_day * exp_cfg.horizon_days
        self.rng = np.random.default_rng(grid_cfg.random_seed)

        self._soc_kwh = grid_cfg.initial_soc * grid_cfg.battery_capacity_kwh
        self._demand: np.ndarray | None = None
        self._renewables: np.ndarray | None = None
        self._price_signal: np.ndarray | None = None
        self._emission_signal: np.ndarray | None = None
        self._step = 0

        self._trace_grid_kw: List[float] = []
        self._trace_unserved_renewables_kw: List[float] = []
        self._trace_battery_throughput: float = 0.0
        self._comfort_violations = 0
        self._total_cost = 0.0
        self._total_emissions = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step = 0
        self._soc_kwh = self.grid_cfg.initial_soc * self.grid_cfg.battery_capacity_kwh
        self._trace_grid_kw = []
        self._trace_unserved_renewables_kw = []
        self._trace_battery_throughput = 0.0
        self._comfort_violations = 0
        self._total_cost = 0.0
        self._total_emissions = 0.0

        self._demand, self._renewables = self._generate_profiles()
        self._price_signal = self._generate_price_signal()
        self._emission_signal = self._generate_emission_signal()

        return self._get_observation()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Run one step.

        Action convention:
        -1.0 means max charging power, +1.0 means max discharging power.
        """

        assert self._demand is not None
        assert self._renewables is not None
        assert self._price_signal is not None
        assert self._emission_signal is not None

        clipped_action = float(np.clip(action, -1.0, 1.0))
        power_setpoint_kw = clipped_action * self.grid_cfg.max_battery_power_kw

        eta_charge = self.grid_cfg.charge_efficiency
        eta_discharge = self.grid_cfg.discharge_efficiency

        max_discharge_kwh = self._soc_kwh
        max_charge_kwh = self.grid_cfg.battery_capacity_kwh - self._soc_kwh

        if power_setpoint_kw >= 0.0:
            # Discharging: SOC drops by raw amount, grid receives eta * amount
            battery_out_kwh = min(power_setpoint_kw * self.grid_cfg.timestep_hours, max_discharge_kwh)
            battery_in_kwh = 0.0
            self._soc_kwh -= battery_out_kwh
            battery_to_grid_kw = (battery_out_kwh * eta_discharge) / self.grid_cfg.timestep_hours
        else:
            # Charging: grid supplies raw amount, only eta * amount is stored
            requested_charge_kwh = -power_setpoint_kw * self.grid_cfg.timestep_hours
            actual_stored_kwh = min(requested_charge_kwh * eta_charge, max_charge_kwh)
            battery_in_kwh = requested_charge_kwh
            battery_out_kwh = 0.0
            self._soc_kwh += actual_stored_kwh
            battery_to_grid_kw = -(battery_in_kwh / self.grid_cfg.timestep_hours)

        demand_kw = self._demand[self._step]
        renewable_kw = self._renewables[self._step]

        net_grid_kw = max(0.0, demand_kw - renewable_kw - battery_to_grid_kw)
        unserved_renewable_kw = max(0.0, renewable_kw + battery_to_grid_kw - demand_kw)

        price = self._price_signal[self._step]
        emissions = self._emission_signal[self._step]

        cost = net_grid_kw * price
        emitted = net_grid_kw * emissions

        comfort_violation = int((self._soc_kwh / self.grid_cfg.battery_capacity_kwh) < self.grid_cfg.soc_comfort_min)
        self._comfort_violations += comfort_violation

        peak_proxy = max(0.0, net_grid_kw - self.exp_cfg.peak_threshold_kw)
        battery_throughput = battery_out_kwh + battery_in_kwh

        reward = -(
            self.exp_cfg.energy_cost_weight * cost
            + self.exp_cfg.emissions_weight * emitted
            + self.exp_cfg.peak_penalty_weight * peak_proxy
            + self.exp_cfg.battery_cycling_weight * battery_throughput
            + self.exp_cfg.comfort_penalty * comfort_violation
        )

        self._trace_grid_kw.append(net_grid_kw)
        self._trace_unserved_renewables_kw.append(unserved_renewable_kw)
        self._trace_battery_throughput += battery_throughput
        self._total_cost += cost
        self._total_emissions += emitted

        self._step += 1
        done = self._step >= self.episode_steps

        info = {
            "grid_kw": net_grid_kw,
            "cost": cost,
            "emissions": emitted,
            "soc": self._soc_kwh / self.grid_cfg.battery_capacity_kwh,
            "comfort_violation": float(comfort_violation),
        }

        obs = self._get_observation() if not done else np.zeros(6, dtype=float)
        return obs, reward, done, info

    def summarize_episode(self) -> EpisodeResult:
        peak = max(self._trace_grid_kw) if self._trace_grid_kw else 0.0
        avg_spill = float(np.mean(self._trace_unserved_renewables_kw)) if self._trace_unserved_renewables_kw else 0.0
        return EpisodeResult(
            total_cost=self._total_cost,
            total_emissions=self._total_emissions,
            peak_demand_kw=peak,
            mean_unserved_renewable_kw=avg_spill,
            comfort_violations=self._comfort_violations,
            battery_throughput_kwh=self._trace_battery_throughput,
        )

    def _get_observation(self) -> np.ndarray:
        assert self._demand is not None
        assert self._renewables is not None
        assert self._price_signal is not None
        assert self._emission_signal is not None

        idx = min(self._step, self.episode_steps - 1)
        hour_of_day = idx % self.steps_per_day
        soc = self._soc_kwh / self.grid_cfg.battery_capacity_kwh

        obs = np.array(
            [
                self._demand[idx] / self.grid_cfg.demand_norm_max,
                self._renewables[idx] / self.grid_cfg.renewable_norm_max,
                self._price_signal[idx],
                self._emission_signal[idx],
                np.sin(2 * np.pi * hour_of_day / self.steps_per_day),
                soc,
            ],
            dtype=float,
        )
        return obs

    def _generate_profiles(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(self.episode_steps)
        hour = t % self.steps_per_day

        base_demand = 70 + 20 * np.sin(2 * np.pi * (hour - 7) / self.steps_per_day)
        evening_peak = 30 * np.exp(-((hour - 18) ** 2) / (2 * 2.3**2))
        noise = self.rng.normal(0.0, self.exp_cfg.demand_noise_std, size=self.episode_steps)
        demand = np.clip(base_demand + evening_peak + noise, 20.0, None)

        solar_shape = np.maximum(0.0, np.sin(2 * np.pi * (hour - 6) / self.steps_per_day))
        renewables = np.clip(100 * self.exp_cfg.renewable_share * solar_shape, 0.0, None)

        return demand, renewables

    def _generate_price_signal(self) -> np.ndarray:
        t = np.arange(self.episode_steps)
        hour = t % self.steps_per_day

        shoulder = 0.12 + 0.03 * np.sin(2 * np.pi * (hour - 12) / self.steps_per_day)
        peak = 0.11 * np.exp(-((hour - 18) ** 2) / (2 * 2.4**2))
        return np.clip(shoulder + peak, 0.05, 0.35)

    def _generate_emission_signal(self) -> np.ndarray:
        t = np.arange(self.episode_steps)
        hour = t % self.steps_per_day

        midday_low = 0.26 - 0.10 * np.exp(-((hour - 13) ** 2) / (2 * 3.2**2))
        evening_high = 0.18 * np.exp(-((hour - 19) ** 2) / (2 * 2.4**2))
        return np.clip(midday_low + evening_high, 0.08, 0.50)
