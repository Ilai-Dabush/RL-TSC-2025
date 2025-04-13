from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic.alias_generators import to_camel

from rltsc.typings.enums import Tags


def sumo_to_snake(value: str) -> str:
    return f"@{to_camel(value)}"

class DetectorStats(BaseModel):
    begin: float
    end: float
    halting_duration_sum: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    id: str
    interval_halting_duration_sum: float
    jam_length_in_meters_sum: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    jam_length_in_vehicles_sum: int = Field(..., json_schema_extra={"tag": Tags.METRICS})
    max_halting_duration: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    max_interval_halting_duration: float
    max_jam_length_in_meters: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    max_jam_length_in_vehicles: int = Field(..., json_schema_extra={"tag": Tags.METRICS})
    max_occupancy: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    max_vehicle_number: int = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_halting_duration: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_interval_halting_duration: float
    mean_max_jam_length_in_meters: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_max_jam_length_in_vehicles: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_occupancy: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_speed: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_time_loss: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    mean_vehicle_number: float = Field(..., json_schema_extra={"tag": Tags.METRICS})
    n_veh_entered: int = Field(..., json_schema_extra={"tag": Tags.METRICS})
    n_veh_left: int
    n_veh_seen: int
    sampled_seconds: float
    started_halts: float = Field(..., json_schema_extra={"tag": Tags.METRICS})

    @computed_field(repr=True)
    @property
    def direction(self) -> str:
        mapping = {
            "_w": "west",
            "_e": "east",
            "_s": "south",
            "_n": "north",
        }

        for k, direction in mapping.items():
            if k in self.id:
                return direction

        raise Exception("Unknown direction")


    model_config = ConfigDict(alias_generator=sumo_to_snake, extra="allow")

class DetectorTag(BaseModel):
    lanes_data: list[DetectorStats] = Field(alias="interval")

    model_config = ConfigDict(extra="ignore")


class IntersectionDetectorsFile(BaseModel):
    detector: DetectorTag
