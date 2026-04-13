from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import GridEnvironment
from smartgrid.policies import RuleBasedPeakShavingPolicy, ZeroActionPolicy


def run_episode(env: GridEnvironment, policy) -> float:
    obs = env.reset(seed=123)
    done = False
    total_reward = 0.0

    while not done:
        obs, reward, done, _ = env.step(policy.act(obs))
        total_reward += reward

    return total_reward


def test_environment_episode_runs_to_completion() -> None:
    env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=2))
    reward = run_episode(env, ZeroActionPolicy())
    summary = env.summarize_episode()

    assert isinstance(reward, float)
    assert summary.peak_demand_kw > 0.0


def test_rule_based_policy_improves_peak() -> None:
    env = GridEnvironment(GridConfig(), ExperimentConfig(horizon_days=3))

    run_episode(env, ZeroActionPolicy())
    baseline = env.summarize_episode()

    run_episode(env, RuleBasedPeakShavingPolicy())
    candidate = env.summarize_episode()

    assert candidate.peak_demand_kw <= baseline.peak_demand_kw
