from pathlib import Path
import xmltodict
import matplotlib.pyplot as plt

from typings.metrics import IntersectionDetectorsFile


class IntersectionMetrics:
    def __init__(self, intersection_xml_data_path: Path):
        self._path = intersection_xml_data_path
        with open(self._path, 'r') as xml_file:
            xml_content = xml_file.read()
            raw_dict = xmltodict.parse(xml_content)
            self._data = IntersectionDetectorsFile.model_validate(raw_dict)

    @property
    def data(self):
        return self._data


    def get_intersection_metrics(self) -> dict:
        pass

    def save_metrics(self) -> None:
        pass


metrics = IntersectionMetrics(Path("../routes/intersection_detectors.xml"))
print(metrics.data)