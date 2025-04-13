from enum import StrEnum


class Tags(StrEnum):
    METRICS = "metrics"


class Metrics(StrEnum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"