from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .environment import GridEnvironment
from .policies import LinearPolicy


@dataclass
class TrainingHistory:
    episode_rewards: List[float]


# ---------------------------------------------------------------------------
# Linear policy (finite-difference)
# ---------------------------------------------------------------------------

def _rollout(env: GridEnvironment, policy: LinearPolicy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    cumulative_reward = 0.0

    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward

    return cumulative_reward


def train_linear_policy(
    env: GridEnvironment,
    episodes: int = 80,
    lr: float = 0.08,
    noise_std: float = 0.10,
) -> tuple[LinearPolicy, TrainingHistory]:
    """Policy-search baseline using finite differences over a linear policy."""

    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 0.12, size=6)
    bias = 0.0
    policy = LinearPolicy(weights=weights, bias=bias)
    history: List[float] = []

    for episode in range(episodes):
        seed = 1000 + episode
        reward_center = _rollout(env, policy, seed)

        grad_w = np.zeros_like(weights)
        grad_b = 0.0

        for _ in range(4):
            delta_w = rng.normal(0.0, noise_std, size=weights.shape)
            delta_b = float(rng.normal(0.0, noise_std))

            plus  = LinearPolicy(weights=weights + delta_w, bias=bias + delta_b)
            minus = LinearPolicy(weights=weights - delta_w, bias=bias - delta_b)

            reward_plus  = _rollout(env, plus,  seed)
            reward_minus = _rollout(env, minus, seed)
            scale = (reward_plus - reward_minus) / 2.0

            grad_w += scale * delta_w
            grad_b += scale * delta_b

        grad_w /= 4.0
        grad_b /= 4.0

        weights += lr * grad_w / (np.linalg.norm(grad_w) + 1e-6)
        bias    += lr * grad_b / (abs(grad_b) + 1e-6)
        policy   = LinearPolicy(weights=weights.copy(), bias=bias)
        history.append(reward_center)

    return policy, TrainingHistory(episode_rewards=history)


# ---------------------------------------------------------------------------
# PPO training via Stable-Baselines3
# ---------------------------------------------------------------------------

def train_ppo_policy(
    env: GridEnvironment,
    total_timesteps: int = 300_000,
    seed: int = 42,
) -> tuple[Any, TrainingHistory]:
    """Train a PPO agent on a GridEnvironment.

    Parameters
    ----------
    env:
        A GridEnvironment already configured with the desired reward weights.
        GridEnvironment is now a gym.Env so it can be passed to SB3 directly.
    total_timesteps:
        Training budget. 300k is enough for meaningful convergence on this
        environment. Use 1_000_000 for results closer to the 15-20% target.
    """
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, TrainingHistory(episode_rewards=[])