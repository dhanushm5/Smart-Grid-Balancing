from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BasePolicy:
    def act(self, observation: np.ndarray) -> float:
        raise NotImplementedError


class ZeroActionPolicy(BasePolicy):
    """No flexibility dispatch: equivalent to not controlling battery."""

    def act(self, observation: np.ndarray) -> float:
        return 0.0


class RuleBasedPeakShavingPolicy(BasePolicy):
    """Simple heuristic: charge on low-carbon midday, discharge on evening peaks."""

    def act(self, observation: np.ndarray) -> float:
        demand_norm, renewable_norm, price, emissions, hour_sin, soc = observation

        # Priority 1: shave high demand periods.
        if demand_norm > 0.72 and soc > 0.12:
            return 0.95

        # Priority 2: absorb renewables or cheap/low-carbon energy when demand is low.
        if demand_norm < 0.52 and soc < 0.92:
            if renewable_norm > 0.45:
                return -0.85
            if emissions < 0.20 and price < 0.14:
                return -0.65

        # Mild charging window (typically midday shoulder) only when demand is not high.
        if hour_sin > 0.50 and demand_norm < 0.56 and soc < 0.85:
            return -0.35

        # Late fallback: prevent very low SOC from persisting, but avoid charging into peaks.
        if soc < 0.10 and demand_norm < 0.50:
            return -0.40

        return 0.0


@dataclass
class LinearPolicy(BasePolicy):
    """Tiny differentiable policy used for a lightweight RL-like training loop."""

    weights: np.ndarray
    bias: float

    def act(self, observation: np.ndarray) -> float:
        val = float(observation @ self.weights + self.bias)
        return float(np.tanh(val))
