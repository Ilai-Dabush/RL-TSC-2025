from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xmltodict
import matplotlib.pyplot as plt

from typings.enums import Metrics
from typings.metrics import IntersectionDetectorsFile, DetectorStats

data = {'@begin': '0.00', '@end': '60.00', '@haltingDurationSum': '26.10', '@id': 'det_n1_0',
        '@intervalHaltingDurationSum': '26.10', '@jamLengthInMetersSum': '1235.00', '@jamLengthInVehiclesSum': '247',
        '@maxHaltingDuration': '25.70', '@maxIntervalHaltingDuration': '25.70', '@maxJamLengthInMeters': '5.00',
        '@maxJamLengthInVehicles': '1', '@maxOccupancy': '10.00', '@maxVehicleNumber': '2',
        '@meanHaltingDuration': '13.05', '@meanIntervalHaltingDuration': '13.05', '@meanMaxJamLengthInMeters': '2.06',
        '@meanMaxJamLengthInVehicles': '0.41', '@meanOccupancy': '3.40', '@meanSpeed': '3.49', '@meanTimeLoss': '9.95',
        '@meanVehicleNumber': '0.71', '@nVehEntered': '4', '@nVehLeft': '3', '@nVehSeen': '4',
        '@sampledSeconds': '42.27', '@startedHalts': '2.00'}


def get_metrics_fields() -> list[str]:
    return [field_name for field_name, field in DetectorStats.model_fields.items() if
            (field.json_schema_extra.get("tag") if field.json_schema_extra else False)]


class IntersectionMetrics:
    def __init__(self, intersection_xml_data_path: Path):
        self._path = intersection_xml_data_path
        with open(self._path, 'r') as xml_file:
            xml_content = xml_file.read()
            raw_dict = xmltodict.parse(xml_content)
            self._data = IntersectionDetectorsFile.model_validate(raw_dict)

    @property
    def metrics(self) -> dict[Metrics, Callable]:
        return {
            Metrics.MAX: np.max,
            Metrics.MIN: np.min,
            Metrics.MEAN: np.mean,
        }


    @property
    def raw_data(self) -> list[DetectorStats]:
        return self._data.detector.lanes_data

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame([x.model_dump(by_alias=False) for x in self.raw_data])


    def get_intersection_metrics(self):
        lane_ids = self.data["id"].unique()
        for lane_id in lane_ids:
            lane_data = self.data[self.data["id"] == lane_id]
            for attribute in get_metrics_fields():
                for metric, f in self.metrics.items():
                    stats = f(lane_data[attribute])

    def save_metrics(self) -> None:
        pass


metrics = IntersectionMetrics(Path("../routes/intersection_detectors.xml"))
print(metrics.data)