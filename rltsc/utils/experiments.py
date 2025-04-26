from pathlib import Path
from typing import Any

from rltsc.config import get_experiment_path_by_name
from rltsc.utils.fs import write_to_yaml_file, read_from_yaml_file


def create_new_training_file_from_existing(experiment_name: str, new_experiment_name: str, new_values: dict[str, Any]) -> None:
    experiment_path = get_experiment_path_by_name(experiment_name)
    experiment = read_from_yaml_file(experiment_path)
    for key, new_value in new_values.items():
        experiment[key] = new_value
    new_experiment_path = Path(str(experiment_path).replace(experiment_name,  new_experiment_name))
    write_to_yaml_file(experiment, new_experiment_path)
