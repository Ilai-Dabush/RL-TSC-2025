import json
from pathlib import Path
from typing import List

from pydantic import TypeAdapter

from typings.experiments import Experiment


def read_config(file_path: Path = Path('experiment_configurations/base_config.json')) -> List[Experiment]:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if not str(file_path).endswith('.json'):
        raise ValueError(f"File {file_path} is not a JSON file.")
    with open(file_path, 'r') as file:
        return TypeAdapter(List[Experiment]).validate_python(json.load(file)["experiments"])
