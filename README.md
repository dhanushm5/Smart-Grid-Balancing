# Smart Grid Balancing

This repository now includes an executable MVP for the project proposal:
an autonomous load-shifting controller that dispatches a battery to reduce
peak demand, cost, and emissions.

## What Is Implemented

- A lightweight smart-grid simulation environment (`smartgrid/environment.py`)
- A no-control baseline and rule-based peak-shaving controller (`smartgrid/policies.py`)
- A trainable linear policy with a simple policy-search loop (`smartgrid/train.py`)
- Stable-Baselines3 PPO agent training for Green vs Economic reward balancing
- A CLI experiment runner with side-by-side comparisons (`scripts/run_experiment.py`)
- A CityLearn backend that can pull a real CityLearn challenge dataset and run a heuristic controller
- Basic tests for environment execution and policy performance (`tests/test_environment.py`)

The current implementation is designed as a practical bridge to your
CityLearn-based target system. It lets you validate reward shaping and compare
control strategies before integrating external simulators.

## MVP Scope Adjustment

**Note for Presentations:** The original project plan proposed an action space encompassing battery dispatch, HVAC setpoint control, and appliance cycle delay. For this MVP, the action space has been reduced to **battery and EV charge/discharge only**. This scope reduction allows for faster validation of the reinforcement learning infrastructure (`Gymnasium` environment wrapping, reward shaping) before introducing complex multi-dimensional actions. If presenting this MVP, be sure to update the PPT deck to reflect this focused action space.

## Quick Start

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the experiment:

```bash
python -m scripts.run_experiment --days 14 --train-episodes 60 --eval-episodes 5
```

To run against the real CityLearn environment instead of the synthetic simulator:

```bash
python -m scripts.run_experiment --backend citylearn --eval-episodes 3
```

The first CityLearn run will clone the upstream CityLearn dataset into a local cache under `~/.cache/smartgrid_balancing/citylearn/` unless you pass `--schema` to point at an existing `schema.json` file.

You should see metrics for:

- `No-Control Baseline`
- `Rule-based policy`
- `Learned linear policy`
- `PPO Green Agent (Emissions Focus)`
- `PPO Economic Agent (Cost Focus)`

with percentage changes in peak demand, cost, and emissions.

**Note on the Learned Linear Policy:** The simple linear policy serves as a lightweight training loop placeholder. While it consistently achieves strong peak demand reductions, you will notice it does so via brute-force over-cycling of the battery, which actually *increases* overall cost and emissions. It is included to demonstrate the policy search functionality, but for a true all-round agent, use the deep PPO models.

## Run Tests

```bash
python -m pytest -q
```

## Project Layout

- `smartgrid/config.py`: simulation and experiment configuration dataclasses
- `smartgrid/environment.py`: grid dynamics, reward, and episode summaries
- `smartgrid/policies.py`: baseline and controllable policies
- `smartgrid/train.py`: lightweight RL-style policy search for linear policy
- `scripts/run_experiment.py`: CLI for training/evaluation and report output
- `smartgrid/citylearn_support.py`: CityLearn dataset bootstrap, environment setup, and heuristic control
- `tests/test_environment.py`: smoke tests and baseline-vs-rule checks

## Next Step Toward CityLearn

After validating this MVP, replace the synthetic profile generation with
CityLearn scenarios while preserving the same policy API and evaluation script.
This keeps your experimentation workflow stable while scaling realism.
