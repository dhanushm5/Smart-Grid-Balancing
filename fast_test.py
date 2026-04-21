from smartgrid.config import ExperimentConfig, GridConfig
from smartgrid.environment import GridEnvironment
from smartgrid.policies import ZeroActionPolicy
from scripts.run_experiment import evaluate_policy, evaluate_ppo
from smartgrid.train import train_ppo_policy

def main():
    grid_cfg = GridConfig()
    base_cfg = ExperimentConfig(horizon_days=14, num_eval_episodes=1)
    env = GridEnvironment(grid_cfg, base_cfg)
    baseline = evaluate_policy(env, ZeroActionPolicy(), 1, seed_base=50)
    
    print("Baseline Peak:", baseline.peak_demand_kw)
    
    # Economic Agent setup
    exp_cfg_econ = ExperimentConfig(
        horizon_days=14,
        energy_cost_weight=3.0,
        emissions_weight=0.0,
        peak_penalty_weight=1.3,
        battery_cycling_weight=0.08,
        comfort_penalty=200.0,
        peak_threshold_kw=90.0,
    )
    env_econ = GridEnvironment(grid_cfg, exp_cfg_econ)
    
    timesteps = 250_000
    print(f"Training PPO for {timesteps} steps...")
    
    ppo_econ_model, _ = train_ppo_policy(env_econ, total_timesteps=timesteps)
    ppo_econ = evaluate_ppo(ppo_econ_model, env_econ, 1, seed_base=50)
    
    reduction = (ppo_econ.peak_demand_kw - baseline.peak_demand_kw) / baseline.peak_demand_kw * 100
    print(f"Econ Peak: {ppo_econ.peak_demand_kw:.2f} ({reduction:.2f}%)")
    print(f"Econ Comfort Violations: {ppo_econ.comfort_violations}")
    print(f"Econ Throughput: {ppo_econ.battery_throughput_kwh:.2f}")

if __name__ == "__main__":
    main()
