import numpy as np
import numpy.typing as npt
import traci
from gymnasium import spaces
from sumo_rl import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction


class FullObservationFunction(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> npt.NDArray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        speeds = self._get_speeds()
        observation = np.array(phase_id + min_green + density + queue + speeds, dtype=np.float32)
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
        print(routes)
        if len(vehicles_ids) == 0:
            return [1e-8 for r in routes]
        # Max speed same for all vehicles
        max_speed = self.ts.sumo.vehicle.getAllowedSpeed(vehicles_ids[0])
        routes_to_vehicles_speeds = {r: [] for r in routes}
        for vehicle_id in vehicles_ids:
            vehicle_route = self.ts.sumo.vehicle.getRouteID(vehicle_id)
            routes_to_vehicles_speeds[vehicle_route].append(self.ts.sumo.vehicle.getSpeed(vehicle_id))

        speeds = []
        for vehicles_speeds in routes_to_vehicles_speeds.values():
            mean = np.mean(vehicles_speeds) / max_speed if vehicles_speeds is not [] else 1e-8
            speeds.append(mean)

        return speeds

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 8 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 8 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
