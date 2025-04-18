import json
import os
from typing import Any

import pandas as pd

from rltsc.config import get_experiment_path_by_name, read_config

MEAN_REWARD_KEY = "env_runners/episode_reward_mean"

def find_progress_files(root_dir: str) -> list[str]:
    progress_files = []
    for root, dirs, files in os.walk(root_dir):
        if "progress.csv" in files:
            progress_files.append(os.path.join(root, "progress.csv"))
    return progress_files

def load_trial_metadata(trial_dir: str) -> dict[str, Any]:
    config_path = os.path.join(trial_dir, "params.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def extract_best_hyperparams(best_config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    # Load the experiment YAML (which includes the param_space)
    experiment = read_config(get_experiment_path_by_name(experiment_name))

    # Extract the tunable hyperparameter names
    param_space = experiment.param_space.model_dump()
    tunable_keys = param_space.keys()

    # Pull their values from the best config
    extracted = {key: best_config.get(key) for key in tunable_keys if key in best_config}

    return extracted

def evaluate_trials(experiment_dir: str, experiment_config_path: str) -> dict[str, Any]:
    progress_files = find_progress_files(experiment_dir)
    best_reward = float('-inf')
    best_config = None
    best_trial = None

    for progress_file in progress_files:
        df = pd.read_csv(progress_file)
        if MEAN_REWARD_KEY not in df.columns:
            continue

        max_reward = df[MEAN_REWARD_KEY].max()
        trial_dir = os.path.dirname(progress_file)
        config = load_trial_metadata(trial_dir)

        print(f"Trial: {os.path.basename(trial_dir)}, Max Reward: {max_reward}")

        if max_reward > best_reward:
            best_reward = max_reward
            best_config = config
            best_trial = os.path.basename(trial_dir)
    best_hp = extract_best_hyperparams(best_config, experiment_config_path)
    print("\n🎯 Best Trial:")
    print(f"Trial: {best_trial}")
    print(f"Max Reward: {best_reward}")
    print("Hyperparameters:")
    print(json.dumps(best_hp, indent=4))
    return {"hp": best_hp, "best_reward": best_reward, "best_trial": best_trial, "best_config": best_config}

if __name__ == "__main__":
    experiment_path = r"C:\Users\ilai\Documents\colman research\experiments_5\DQN_SingleAgent\DQN_2025-04-17_19-17-37"
    experiment_config = "DQN"
    evaluate_trials(experiment_path, experiment_config)
