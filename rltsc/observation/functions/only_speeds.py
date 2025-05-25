import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from sumo_rl import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction


class SpeedsObservation(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> npt.NDArray:
        """Return the default observation."""
        speeds = self._get_speeds()
        observation = np.array(speeds, dtype=np.float32)
        return observation

    def _get_vehicles(self) -> list:
        veh_list = []
        for lane in self.ts.lanes:
            veh_list += self.ts.sumo.lane.getLastStepVehicleIDs(lane)

        return veh_list

    def _get_speeds(self) -> list[float]:
        """
        Gets avg speed per direction
        :return: list of speeds
        """
        vehicles_ids = self._get_vehicles()
        routes = self.ts.sumo.route.getIDList()
        if len(vehicles_ids) == 0:
            return [0 for r in routes]
        # Max speed same for all vehicles
        max_speed = 60.0
        routes_to_vehicles_speeds = {r: [] for r in routes}
        for vehicle_id in vehicles_ids:
            vehicle_route = self.ts.sumo.vehicle.getRouteID(vehicle_id)
            routes_to_vehicles_speeds[vehicle_route].append(self.ts.sumo.vehicle.getSpeed(vehicle_id))

        speeds = []
        for vehicles_speeds in routes_to_vehicles_speeds.values():
            clean_speeds = [s for s in vehicles_speeds if s and not np.isnan(s)]
            mean = np.mean(clean_speeds) / max_speed if clean_speeds else 0
            speeds.append(mean if not np.isnan(mean) else 0)

        return speeds

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros( 8, dtype=np.float32),
            high=np.ones(8 , dtype=np.float32),
        )
