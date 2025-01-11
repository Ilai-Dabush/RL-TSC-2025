from ray.rllib.env import EnvContext
from sumo_rl import SumoEnvironment


def env_creator(env_config: EnvContext):
    env = SumoEnvironment(
        net_file="/content/sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="/content/single-intersection-vhvh.rou.xml",
        out_csv_name="/content/drive/MyDrive/Colman Research 2024-2025/results/single-2way/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=3000,
        yellow_time=4,
        min_green=5,
        max_green=60,
        reward_fn="pressure",
        add_system_info=True
    )
    return CustomObservationWrapper(env)


env_name = "DDQN_single"
register_env(env_name, env_creator)
