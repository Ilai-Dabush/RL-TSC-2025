import os
import sys

from rltsc.typings.enums import Platforms
from rltsc.utils.os_utils import get_platform


def bootstrap():
    os.environ["SUMO_HOME"] = (
        r"C:\Program Files (x86)\Eclipse\Sumo"
        if get_platform() == Platforms.WINDOWS
        else "/usr/share/sumo"
    )
    os.environ["LIBSUMO_AS_TRACI"] = "1"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
