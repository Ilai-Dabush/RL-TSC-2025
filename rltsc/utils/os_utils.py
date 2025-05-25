import platform
from typing import cast

from rltsc.typings.enums import Platforms


def get_platform() -> Platforms:
    return cast(Platforms, platform.system())
