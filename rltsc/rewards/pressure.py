import numpy as np
from sumo_rl import TrafficSignal


def normalize_to_0_to_n(value: float, n: int = 100):
    normalized = 1 / (1 + np.exp(-value))
    return normalized * n


def normalized_pressure(traffic_signal: TrafficSignal):
    return -normalize_to_0_to_n(traffic_signal.get_pressure())


def pressure_clip(traffic_signal: TrafficSignal) -> float:
    pressure_rwd = traffic_signal.get_pressure()
    return max(-100, pressure_rwd)


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