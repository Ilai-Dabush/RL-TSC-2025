from enum import Enum


class Tags(str, Enum):
    METRICS = "metrics"


class Metrics(str, Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"