"""Diagnostic: print CityLearn observation names, value ranges, and action spaces."""
from smartgrid.citylearn_support import build_citylearn_env, CityLearnBackendConfig
import numpy as np

cfg = CityLearnBackendConfig(episode_time_steps=48)
env = build_citylearn_env(cfg)

print("=== Action Space ===")
for i, space in enumerate(env.action_space):
    print(f"  Agent {i}: low={space.low}, high={space.high}")
print(f"  Action names: {env.action_names}")

print("\n=== Observation Space ===")
print(f"  Observation names: {env.observation_names}")

obs, _ = env.reset()
print(f"\n=== Initial Observation (agent 0, first 48 steps) ===")
print(f"  Shape: {np.array(obs[0]).shape}")
for name, val in zip(env.observation_names[0], obs[0]):
    print(f"  {name:40s} = {val:.4f}")

# Step a few times to see how values evolve
print("\n=== Observation samples at steps 6, 12, 18 ===")
for target in [6, 12, 18]:
    for _ in range(6):
        zero_actions = [np.zeros(env.action_space[0].shape[0]).tolist()]
        obs, _, term, trunc, _ = env.step(zero_actions)
        if term or trunc:
            break
    print(f"\n  Step ~{target}:")
    for name, val in zip(env.observation_names[0], obs[0]):
        print(f"    {name:40s} = {val:.4f}")

env.close()
