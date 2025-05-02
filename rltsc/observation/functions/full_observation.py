import numpy as np
import numpy.typing as npt
import traci
from sumo_rl.environment.observations import DefaultObservationFunction


class FullObservationFunction(DefaultObservationFunction):

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
        if len(vehicles_ids) == 0:
            return []
        # Max speed same for all vehicles
        max_speed = self.ts.sumo.vehicle.getAllowedSpeed(vehicles_ids[0])
        routes_to_vehicles_speeds = {}
        for vehicle_id in vehicles_ids:
            vehicle_route = traci.vehicle.getRouteID(vehicle_id)
            if vehicle_route not in routes_to_vehicles_speeds:
                routes_to_vehicles_speeds[vehicle_route] = []
            routes_to_vehicles_speeds[vehicle_route].append(self.ts.sumo.vehicle.getSpeed(vehicle_id))

        speeds = []

        for vehicles_speeds in routes_to_vehicles_speeds.values():
            speeds.append(np.mean(vehicles_speeds)/max_speed)

        return speeds
