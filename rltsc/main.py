import os
import platform
import sys

import ray

from rltsc.config import read_config, get_experiment_path_by_name
from rltsc.utils.experiments import create_new_training_file_from_existing
from rltsc.utils.sumo import fit


def set_env() -> None:
    os.environ["SUMO_HOME"] = (
        r"C:\Program Files (x86)\Eclipse\Sumo"
        if platform.system() == "Windows"
        else "/usr/share/sumo"
    )
    os.environ["LIBSUMO_AS_TRACI"] = "1"
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

def run(experiment_name: str) -> None:
    ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    set_env()
    experiment = read_config(get_experiment_path_by_name(experiment_name))
    fit(experiment)



if __name__ == "__main__":
    run("DQN")
