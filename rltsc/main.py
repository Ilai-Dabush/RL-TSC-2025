import sys
import platform
import os

import ray

from rltsc.utils import read_config
from rltsc.utils.sumo import fit
from rltsc.utils.utils import get_experiment_path_by_name
from ray.autoscaler.sdk import request_resources


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
    ray.init(num_cpus=4, num_gpus=2)
    set_env()
    experiment = read_config(get_experiment_path_by_name(experiment_name))
    fit(experiment)



if __name__ == "__main__":
    run("DQN")
