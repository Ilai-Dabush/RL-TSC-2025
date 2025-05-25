from pathlib import Path
from typing import Any

import yaml


def write_to_yaml_file(content: Any, file_path: Path) -> None:
    with open(file_path, "w") as f:
        yaml.dump(content, f)


def read_from_yaml_file(file_path: Path) -> Any:
    with open(file_path, "r") as f:
        return yaml.safe_load(f.read())
