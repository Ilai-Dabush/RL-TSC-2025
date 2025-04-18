from pathlib import Path

import yaml

from rltsc.typings.experiments import Experiment


def get_experiment_path_by_name(name: str) -> Path:
    return Path(f"rltsc/experiment_configurations/{name}.yaml")


def read_config(
        file_path: Path
) -> Experiment:
    with open(file_path, "r") as file:
        experiment_data = yaml.safe_load(file.read())
        return Experiment.model_validate(experiment_data)