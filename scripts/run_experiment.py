from __future__ import annotations

import argparse
from statistics import mean

from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import EpisodeResult, GridEnvironment
from smartgrid.policies import BasePolicy, RuleBasedPeakShavingPolicy, ZeroActionPolicy
from smartgrid.train import train_linear_policy


def evaluate_policy(env: GridEnvironment, policy: BasePolicy, num_episodes: int, seed_base: int) -> EpisodeResult:
    results: list[EpisodeResult] = []

    for i in range(num_episodes):
        obs = env.reset(seed=seed_base + i)
        done = False

        while not done:
            action = policy.act(obs)
            obs, _, done, _ = env.step(action)

        results.append(env.summarize_episode())

    return EpisodeResult(
        total_cost=mean([r.total_cost for r in results]),
        total_emissions=mean([r.total_emissions for r in results]),
        peak_demand_kw=mean([r.peak_demand_kw for r in results]),
        mean_unserved_renewable_kw=mean([r.mean_unserved_renewable_kw for r in results]),
        comfort_violations=int(mean([r.comfort_violations for r in results])),
        battery_throughput_kwh=mean([r.battery_throughput_kwh for r in results]),
    )


def print_comparison(name: str, baseline: EpisodeResult, candidate: EpisodeResult) -> None:
    def pct_delta(new: float, old: float) -> float:
        if old == 0:
            return 0.0
        return (new - old) / old * 100.0

    print(f"\n=== {name} vs No-Control Baseline ===")
    print(f"Peak demand (kW):       {candidate.peak_demand_kw:7.2f}  ({pct_delta(candidate.peak_demand_kw, baseline.peak_demand_kw):+6.2f}%)")
    print(f"Total cost (arbitrary): {candidate.total_cost:7.2f}  ({pct_delta(candidate.total_cost, baseline.total_cost):+6.2f}%)")
    print(f"Total emissions:        {candidate.total_emissions:7.2f}  ({pct_delta(candidate.total_emissions, baseline.total_emissions):+6.2f}%)")
    print(f"Battery throughput:     {candidate.battery_throughput_kwh:7.2f}  ({pct_delta(candidate.battery_throughput_kwh, baseline.battery_throughput_kwh):+6.2f}%)")
    print(f"Comfort violations:     {candidate.comfort_violations}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart-grid load-shifting experiment")
    parser.add_argument("--days", type=int, default=14, help="Episode horizon in days")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--train-episodes", type=int, default=60, help="Training episodes for learned policy")
    args = parser.parse_args()

    grid_cfg = GridConfig()
    exp_cfg = ExperimentConfig(horizon_days=args.days, num_eval_episodes=args.eval_episodes)
    env = GridEnvironment(grid_cfg, exp_cfg)

    baseline = evaluate_policy(env, ZeroActionPolicy(), args.eval_episodes, seed_base=50)
    rule_based = evaluate_policy(env, RuleBasedPeakShavingPolicy(), args.eval_episodes, seed_base=50)

    learned_policy, history = train_linear_policy(env, episodes=args.train_episodes)
    learned = evaluate_policy(env, learned_policy, args.eval_episodes, seed_base=150)

    print("Smart Grid Balancing Experiment")
    print("--------------------------------")
    print(f"Training episodes: {args.train_episodes}")
    print(f"Final training reward: {history.episode_rewards[-1]:.2f}")

    print_comparison("Rule-based policy", baseline, rule_based)
    print_comparison("Learned linear policy", baseline, learned)


if __name__ == "__main__":
    main()
