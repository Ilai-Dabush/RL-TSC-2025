import json
from pathlib import Path


def read_config(file_path: Path = Path('experiment_configurations/base_config.json')):
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if not str(file_path).endswith('.json'):
        raise ValueError(f"File {file_path} is not a JSON file.")
    with open(file_path, 'r') as file:
        return json.load(file)
