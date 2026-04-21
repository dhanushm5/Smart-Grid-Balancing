from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import EpisodeResult, GridEnvironment
from smartgrid.policies import BasePolicy, RuleBasedPeakShavingPolicy, ZeroActionPolicy
from smartgrid.train import train_linear_policy, train_ppo_policy
from smartgrid.citylearn_support import CityLearnBackendConfig, evaluate_citylearn_controller


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    env: GridEnvironment,
    policy: BasePolicy,
    num_episodes: int,
    seed_base: int,
) -> EpisodeResult:
    """Evaluate a rule-based or linear policy on a GridEnvironment."""
    results: list[EpisodeResult] = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append(env.summarize_episode())

    return _average_results(results)


def evaluate_ppo(
    model: Any,
    env: GridEnvironment,
    num_episodes: int,
    seed_base: int,
) -> EpisodeResult:
    """Evaluate a trained SB3 PPO model.

    IMPORTANT: always pass the same env the model was trained on so the
    reward weights (and therefore the metrics) are consistent.
    """
    results: list[EpisodeResult] = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            action, _ = model.predict(obs.astype(np.float32), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append(env.summarize_episode())

    return _average_results(results)


def _average_results(results: list[EpisodeResult]) -> EpisodeResult:
    return EpisodeResult(
        total_cost=mean([r.total_cost for r in results]),
        total_emissions=mean([r.total_emissions for r in results]),
        peak_demand_kw=mean([r.peak_demand_kw for r in results]),
        mean_unserved_renewable_kw=mean([r.mean_unserved_renewable_kw for r in results]),
        comfort_violations=int(mean([r.comfort_violations for r in results])),
        battery_throughput_kwh=mean([r.battery_throughput_kwh for r in results]),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(name: str, baseline: EpisodeResult, candidate: EpisodeResult) -> None:
    def pct(new: float, old: float) -> str:
        return "   N/A" if old == 0 else f"{(new - old) / old * 100.0:+6.2f}%"

    print(f"\n=== {name} vs No-Control Baseline ===")
    print(f"Peak demand (kW):       {candidate.peak_demand_kw:7.2f}  ({pct(candidate.peak_demand_kw, baseline.peak_demand_kw)})")
    print(f"Total cost (arbitrary): {candidate.total_cost:7.2f}  ({pct(candidate.total_cost, baseline.total_cost)})")
    print(f"Total emissions:        {candidate.total_emissions:7.2f}  ({pct(candidate.total_emissions, baseline.total_emissions)})")
    print(f"Battery throughput:     {candidate.battery_throughput_kwh:7.2f}  ({pct(candidate.battery_throughput_kwh, baseline.battery_throughput_kwh)})")
    print(f"Comfort violations:     {candidate.comfort_violations}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smart-grid load-shifting experiment")
    parser.add_argument("--backend", choices=["synthetic", "citylearn"], default="synthetic")
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="citylearn_challenge_2022_phase_all_plus_evs")
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=60,
                        help="Used only for the linear policy finite-difference loop.")
    parser.add_argument("--ppo-timesteps", type=int, default=600_000,
                        help="Total env timesteps for each PPO agent. "
                             "Use 1000000 for full convergence.")
    args = parser.parse_args()

    # ── CityLearn backend ──────────────────────────────────────────────────
    if args.backend == "citylearn":
        cache_root = (
            Path(args.cache_root).expanduser()
            if args.cache_root
            else Path.home() / ".cache" / "smartgrid_balancing" / "citylearn"
        )
        env_config = CityLearnBackendConfig(
            dataset_name=args.dataset_name,
            episode_time_steps=args.days * 24,
            random_seed=0,
            cache_root=cache_root,
            schema_path=Path(args.schema).expanduser() if args.schema else None,
        )
        baseline, candidate = evaluate_citylearn_controller(env_config, episodes=args.eval_episodes)
        print("Smart Grid Balancing Experiment")
        print("--------------------------------")
        print(f"Backend: CityLearn ({env_config.dataset_name})")
        print(f"Episode time steps: {env_config.episode_time_steps}")
        print_comparison("CityLearn heuristic policy", baseline, candidate)
        return

    # ── Synthetic backend ──────────────────────────────────────────────────
    grid_cfg = GridConfig()

    # Base env — default blended reward, used for baselines + linear policy
    base_cfg = ExperimentConfig(horizon_days=args.days, num_eval_episodes=args.eval_episodes)
    env = GridEnvironment(grid_cfg, base_cfg)

    print("Smart Grid Balancing Experiment")
    print("--------------------------------")
    print(f"Training episodes (linear): {args.train_episodes}")
    print(f"PPO timesteps per agent:    {args.ppo_timesteps:,}")

    # No-control baseline
    baseline = evaluate_policy(env, ZeroActionPolicy(), args.eval_episodes, seed_base=50)

    # Rule-based
    rule_based = evaluate_policy(env, RuleBasedPeakShavingPolicy(), args.eval_episodes, seed_base=50)

    # Linear policy
    learned_policy, _ = train_linear_policy(env, episodes=args.train_episodes)
    learned = evaluate_policy(env, learned_policy, args.eval_episodes, seed_base=150)

    # PPO Green — energy_cost_weight=0 (pure carbon signal)
    # Trained AND evaluated on env_green so metrics reflect what it optimised for
    print(f"\nTraining PPO Green agent (emissions only, {args.ppo_timesteps:,} timesteps)...")
    # Green agent — let emissions signal dominate
    exp_cfg_green = ExperimentConfig(
        horizon_days=args.days,
        energy_cost_weight=0.0,
        emissions_weight=3.0,
        peak_penalty_weight=4.0,   # Massive boost over the other signals
        battery_cycling_weight=0.08,
        comfort_penalty=500.0,     # Huge increase to prevent agent from gaming the penalty
    )
    env_green = GridEnvironment(grid_cfg, exp_cfg_green)
    ppo_green_model, _ = train_ppo_policy(env_green, total_timesteps=args.ppo_timesteps)
    ppo_green = evaluate_ppo(ppo_green_model, env_green, args.eval_episodes, seed_base=150)

    # PPO Economic — emissions_weight=0 (pure price signal)
    # Trained AND evaluated on env_econ
    print(f"Training PPO Economic agent (cost only, {args.ppo_timesteps:,} timesteps)...")
    # Economic agent — let price signal dominate  
    exp_cfg_econ = ExperimentConfig(
        horizon_days=args.days,
        energy_cost_weight=3.0,
        emissions_weight=0.0,
        peak_penalty_weight=4.0,   # Massive boost over the other signals
        battery_cycling_weight=0.08,
        comfort_penalty=500.0,     # Huge increase to prevent agent from gaming the penalty
    )
    env_econ = GridEnvironment(grid_cfg, exp_cfg_econ)
    ppo_econ_model, _ = train_ppo_policy(env_econ, total_timesteps=args.ppo_timesteps)
    ppo_econ = evaluate_ppo(ppo_econ_model, env_econ, args.eval_episodes, seed_base=150)

    # Results
    print_comparison("Rule-based policy", baseline, rule_based)
    print_comparison("Learned linear policy", baseline, learned)
    print_comparison("PPO Green Agent (Emissions Focus)", baseline, ppo_green)
    print_comparison("PPO Economic Agent (Cost Focus)", baseline, ppo_econ)

    # Head-to-head summary for the research question
    print("\n--- Green vs Economic: head-to-head ---")
    print(f"Emissions:  Green={ppo_green.total_emissions:.1f}  Econ={ppo_econ.total_emissions:.1f}  "
          f"→ green saves {(ppo_econ.total_emissions - ppo_green.total_emissions) / ppo_econ.total_emissions * 100:.2f}% more")
    print(f"Cost:       Green={ppo_green.total_cost:.1f}  Econ={ppo_econ.total_cost:.1f}  "
          f"→ econ saves {(ppo_green.total_cost - ppo_econ.total_cost) / ppo_green.total_cost * 100:.2f}% more")
    print(f"Throughput: Green={ppo_green.battery_throughput_kwh:.1f}  Econ={ppo_econ.battery_throughput_kwh:.1f}")


if __name__ == "__main__":
    main()