import sys
import platform
import os

from utils import read_config


def set_env():
    os.environ["SUMO_HOME"] = (
        r"C:\Program Files (x86)\Eclipse\Sumo"
        if platform.system() == "Windows"
        else "/usr/share/sumo"
    )
    os.environ["LIBSUMO_AS_TRACI"] = "1"
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)


if __name__ == "__main__":
    set_env()
    c = read_config()
    print(c)
