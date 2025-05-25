from typing import Union, Optional

import gymnasium as gym
import ray
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from typing_extensions import override


class ResourcesCallback(DefaultCallbacks):

    @override
    def on_episode_start(self, *,
                         episode: Union[EpisodeType, EpisodeV2],
                         env_runner: EnvRunner = None,
                         metrics_logger: Optional[MetricsLogger] = None,
                         env: Optional[gym.Env] = None,
                         env_index: int,
                         rl_module: Optional[RLModule] = None,
                         worker: EnvRunner = None,
                         base_env: Optional[BaseEnv] = None,
                         policies: Optional[dict[PolicyID, Policy]] = None,
                         **kwargs
                         ):
        print("Episode", episode)
        print(f'resources - {ray.nodes()}')
