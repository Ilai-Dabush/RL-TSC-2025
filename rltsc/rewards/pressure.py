import time

import numpy as np
import ray
from sumo_rl import TrafficSignal


def normalize_to_0_to_n(value: float, n: int = 100):
    normalized = 1 / (1 + np.exp(-value))
    return normalized * n


def normalized_pressure(traffic_signal: TrafficSignal):
    return -normalize_to_0_to_n(traffic_signal.get_pressure())


def get_route_connections(traffic_signal: TrafficSignal):
    """
    Gets route connections directly from a running SUMO instance using TraCI.
    """
    # Get all route IDs
    route_ids = traffic_signal.sumo.route.getIDList()

    # Create a dictionary to store routes and their edges
    routes = {}
    for route_id in route_ids:
        edges = traffic_signal.sumo.route.getEdges(route_id)
        routes[route_id] = edges

    # Find route connections based on matching edges
    route_connections = {}
    for route_id, edges in routes.items():
        # Find routes that start with the last edge of this route
        connections = []
        last_edge = edges[-1]
        for other_id, other_edges in routes.items():
            if other_id != route_id and other_edges[0] == last_edge:
                connections.append(other_id)

        route_connections[route_id] = connections

    return routes, route_connections


def calculate_route_pressure(traffic_signal: TrafficSignal):
    """
    Calculates the pressure (veh_out - veh_in) for each route in the intersection.
    Pressure is measured as the difference between vehicles that have exited
    the last edge of the route and vehicles that have entered the first edge.

    Returns:
        dict: A dictionary mapping route IDs to their current pressure values
    """
    route_ids = traffic_signal.sumo.route.getIDList()
    routes = {route_id: traffic_signal.sumo.route.getEdges(route_id) for route_id in route_ids}

    # Initialize the pressure dictionary
    pressure = {}

    for route_id, edges in routes.items():
        # We need at least first and last edge to calculate pressure
        if len(edges) >= 2:
            first_edge = edges[0]
            last_edge = edges[-1]

            # Get vehicle counts
            veh_in = traffic_signal.sumo.edge.getLastStepVehicleNumber(first_edge)
            veh_out = traffic_signal.sumo.edge.getLastStepVehicleNumber(last_edge)

            # Calculate pressure
            pressure[route_id] = veh_out - veh_in

    return pressure


def get_edge_occupancy(traffic_signal: TrafficSignal):
    """
    Gets the occupancy percentage for each edge.
    Useful for more detailed traffic analysis.

    Returns:
        dict: A dictionary mapping edge IDs to their current occupancy values
    """
    edge_ids = traffic_signal.sumo.edge.getIDList()
    occupancy = {edge_id: traffic_signal.sumo.edge.getLastStepOccupancy(edge_id) for edge_id in edge_ids}
    return occupancy


def get_advanced_route_pressure(traffic_signal: TrafficSignal):
    """
    An advanced method for calculating route pressure that considers
    halting vehicles, occupancy, and queue length.

    Returns:
        dict: A dictionary with detailed pressure metrics for each route
    """
    route_ids = traffic_signal.sumo.route.getIDList()
    routes = {route_id: traffic_signal.sumo.route.getEdges(route_id) for route_id in route_ids}

    advanced_pressure = {}

    for route_id, edges in routes.items():
        if len(edges) >= 2:
            first_edge = edges[0]
            last_edge = edges[-1]

            # Basic vehicle counts
            veh_in = traffic_signal.sumo.edge.getLastStepVehicleNumber(first_edge)
            veh_out = traffic_signal.sumo.edge.getLastStepVehicleNumber(last_edge)

            # Additional metrics
            halting_in = traffic_signal.sumo.edge.getLastStepHaltingNumber(first_edge)
            halting_out = traffic_signal.sumo.edge.getLastStepHaltingNumber(last_edge)

            occupancy_in = traffic_signal.sumo.edge.getLastStepOccupancy(first_edge)
            occupancy_out = traffic_signal.sumo.edge.getLastStepOccupancy(last_edge)

            # Some SUMO versions have queue length method
            try:
                queue_in = traffic_signal.sumo.edge.getLastStepQueueLength(first_edge)
                queue_out = traffic_signal.sumo.edge.getLastStepQueueLength(last_edge)
            except:
                queue_in = 0
                queue_out = 0

            # Calculate weighted pressure (can be adjusted based on importance)
            basic_pressure = veh_out - veh_in
            queue_pressure = queue_out - queue_in
            occupancy_pressure = occupancy_out - occupancy_in
            halting_pressure = halting_out - halting_in

            advanced_pressure[route_id] = {
                'basic': basic_pressure,
                'queue': queue_pressure,
                'occupancy': occupancy_pressure,
                'halting': halting_pressure,
                'weighted': basic_pressure + 2 * queue_pressure + occupancy_pressure * 100
            }

    return advanced_pressure


def pressure_clip_advanced(clip: int, alpha: float, traffic_signal: TrafficSignal) -> float:
    pressure = calculate_route_pressure(traffic_signal)
    pressures = np.array([np.clip(p_value, -0.05, clip) for p_value in pressure.values()])
    p_mean = np.mean(pressures)
    p_var = np.var(pressures)

    # Optional: normalize variance by mean pressure to make it scale-independent
    normalized_var = p_var / (p_mean + 1e-6)

    reward = p_mean + alpha * normalized_var

    # ray.logger.info("============================================")
    # ray.logger.info(f"""
    #     pressures: {pressures}\n
    #     p_mean: {p_mean}\n
    #     p_var: {p_var}\n
    #     normalized_var: {normalized_var}\n
    #     reward: {reward}\n
    #     """)
    # ray.logger.info("============================================")
    return reward


def pressure_clip(clip: int, traffic_signal: TrafficSignal) -> float:
    pressure_rwd = traffic_signal.get_pressure()
    return max(clip, pressure_rwd*2)


def negative_mean_pressure_avg_and_clipped(ts: TrafficSignal):
    """
    Computes the negative mean pressure per (incoming -> outgoing) flow for a given TrafficSignal object.
    The reward is clipped between 0 and 100.

    Args:
        ts (TrafficSignal): A traffic signal object from SUMO-RL.

    Returns:
        float: The clipped negative mean pressure.
    """
    pressures = []
    for lane in ts.lanes:
        # Get the number of vehicles approaching
        incoming_veh = ts.sumo.lane.getLastStepVehicleNumber(lane)

        # Get all outgoing lanes connected to this incoming lane
        connected_lanes = ts.sumo.lane.getLinks(lane)
        outgoing_veh = 0
        for conn in connected_lanes:
            if conn:  # Ensure it's not empty
                out_lane = conn[0]
                outgoing_veh += ts.sumo.lane.getLastStepVehicleNumber(out_lane)

        pressure = outgoing_veh - incoming_veh
        pressures.append(pressure)

    if pressures:
        mean_pressure = np.mean(pressures)
    else:
        mean_pressure = 0.0

    # Return the negative pressure, clipped from 0 to 100
    reward = -float(np.clip(mean_pressure, 0, 100))
    return reward