from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

data = {'@begin': '0.00', '@end': '60.00', '@haltingDurationSum': '26.10', '@id': 'det_n1_0',
        '@intervalHaltingDurationSum': '26.10', '@jamLengthInMetersSum': '1235.00', '@jamLengthInVehiclesSum': '247',
        '@maxHaltingDuration': '25.70', '@maxIntervalHaltingDuration': '25.70', '@maxJamLengthInMeters': '5.00',
        '@maxJamLengthInVehicles': '1', '@maxOccupancy': '10.00', '@maxVehicleNumber': '2',
        '@meanHaltingDuration': '13.05', '@meanIntervalHaltingDuration': '13.05', '@meanMaxJamLengthInMeters': '2.06',
        '@meanMaxJamLengthInVehicles': '0.41', '@meanOccupancy': '3.40', '@meanSpeed': '3.49', '@meanTimeLoss': '9.95',
        '@meanVehicleNumber': '0.71', '@nVehEntered': '4', '@nVehLeft': '3', '@nVehSeen': '4',
        '@sampledSeconds': '42.27', '@startedHalts': '2.00'}

def sumo_to_snake(value: str) -> str:
    return f"@{to_camel(value)}"

class DetectorStats(BaseModel):
    begin: float
    end: float
    halting_duration_sum: float
    id: str
    interval_halting_duration_sum: float
    jam_length_in_meters_sum: float
    jam_length_in_vehicles_sum: int
    max_halting_duration: float
    max_interval_halting_duration: float
    max_jam_length_in_meters: float
    max_jam_length_in_vehicles: int
    max_occupancy: float
    max_vehicle_number: int
    mean_halting_duration: float
    mean_interval_halting_duration: float
    mean_max_jam_length_in_meters: float
    mean_max_jam_length_in_vehicles: float
    mean_occupancy: float
    mean_speed: float
    mean_time_loss: float
    mean_vehicle_number: float
    n_veh_entered: int
    n_veh_left: int
    n_veh_seen: int
    sampled_seconds: float
    started_halts: float


    model_config = ConfigDict(alias_generator=sumo_to_snake, extra="allow")

class DetectorTag(BaseModel):
    interval: list[DetectorStats]

    model_config = ConfigDict(extra="ignore")


class IntersectionDetectorsFile(BaseModel):
    detector: DetectorTag
