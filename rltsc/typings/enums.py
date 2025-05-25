from enum import Enum


class Tags(str, Enum):
    METRICS = "metrics"


class Metrics(str, Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class Platforms(str, Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"


class AlgorithmNames(str, Enum):
    IDENTITY = "Identity"
    MAHSUP = "MAHSUP"
