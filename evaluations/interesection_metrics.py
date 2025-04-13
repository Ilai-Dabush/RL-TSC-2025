import datetime
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xmltodict

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
            (field.json_schema_extra.get("tag") if hasattr(field, "json_schema_extra") else False)]


class IntersectionMetrics:
    def __init__(self, intersection_xml_data_path: Path, output_csv_path: Path):
        self._path = intersection_xml_data_path
        self._output_csv_path = output_csv_path
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

    @property
    def output_csv_path(self) -> str:
        return rf"{self._output_csv_path}\output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"


    def get_intersection_metrics(self) -> pd.DataFrame:
        direction = self.data["direction"].unique()
        result = []

        for direction in direction:
            lane_data = self.data[self.data["direction"] == direction]
            for attribute in get_metrics_fields():
                stats = {"direction": direction, "attribute": attribute}
                for metric, f in self.metrics.items():
                    stats[metric.value] = f(lane_data[attribute].to_numpy())
                result.append(stats)

        metrics_df = pd.DataFrame(result)
        metrics_df.to_csv(self.output_csv_path, index=False)
        return metrics_df


metrics = IntersectionMetrics(Path("../routes/intersection_detectors.xml"), Path("../outputs"))
metrics.get_intersection_metrics()