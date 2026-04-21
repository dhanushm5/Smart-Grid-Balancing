from smartgrid.citylearn_support import build_citylearn_env, CityLearnBackendConfig
env = build_citylearn_env(CityLearnBackendConfig())
action_names = env.action_names[0]
lows = env.action_space[0].low
highs = env.action_space[0].high
for n, l, h in zip(action_names, lows, highs):
    print(f"{n}: [{l}, {h}]")
