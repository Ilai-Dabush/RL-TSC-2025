from typing import Mapping, Any

import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig, APPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env import EnvContext
from ray.train import RunConfig, CheckpointConfig
from ray.tune import register_env
from sumo_rl import SumoEnvironment

from rltsc.callbacks.debug import DebugCallback
from rltsc.callbacks.resources import ResourcesCallback
from rltsc.config import read_config, get_experiment_path_by_name
from rltsc.typings.algorithms import ALGORITHM_NAMES
from rltsc.typings.experiments import Experiment
from rltsc.utils.sumo import bootstrap
from rltsc.wrappers.gym import CustomObservationWrapper

CONFIG_MAPPER: Mapping[ALGORITHM_NAMES, type[AlgorithmConfig]] = {
    "DQN": DQNConfig,
    "PPO": PPOConfig,
    "APPO": APPOConfig,
    "DDQN": DQNConfig,
}

class Trainer:

    experiment: Experiment

    def __init__(self, experiment_name: str):
        self.experiment = read_config(get_experiment_path_by_name(experiment_name))
        self._bootsrap()

    def _bootsrap(self):
        bootstrap()
        ray.shutdown()
        ray.init(num_cpus=4, num_gpus=self.experiment.num_gpus, ignore_reinit_error=True)

    def create_env(
            self,
    ) -> None:
        def env_creator(env_config: EnvContext):
            env = SumoEnvironment(
                net_file=self.experiment.net_file,
                route_file=self.experiment.rou_file,
                out_csv_name=self.experiment.out_csv_path,
                single_agent=True,
                use_gui=False,
                # num_seconds=20000,
                yellow_time=self.experiment.min_yellow_time,
                min_green=self.experiment.min_green_time,
                reward_fn=self.experiment.reward_fn,
                # reward_fn=experiment.reward_fn,
                add_system_info=True,
            )
            return CustomObservationWrapper(env)

        register_env(self.experiment.experiment_type, env_creator)

    def create_env_with_config(self) -> tuple[AlgorithmConfig, RunConfig]:
        self.create_env()

        config = (
            CONFIG_MAPPER[self.experiment.algo_name]()
            .environment(self.experiment.experiment_type, env_config={"horizon": 10_000})
            .callbacks(ResourcesCallback)
            .callbacks(DebugCallback)
            .env_runners(num_env_runners=self.experiment.num_env_runners, rollout_fragment_length=100)
            # .learners(num_learners=2, num_gpus_per_learner=0.5, num_cpus_per_learner=1)
            .training(**self.experiment.config.model_dump(exclude={"algo_name", "override_num_gpus"}),
                      replay_buffer_config={'type': 'PrioritizedEpisodeReplayBuffer',
                                            "capacity": 50000,
                                            "alpha": 0.6,
                                            # Beta parameter for sampling from prioritized replay buffer.
                                            "beta": 0.4}
            )
            .learners(num_gpus_per_learner=self.experiment.num_gpus, num_cpus_per_learner=1)
            .debugging(log_level=self.experiment.log_level)
            .framework(framework=self.experiment.framework)
            .resources(num_gpus=self.experiment.num_gpus)
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=1,
                evaluation_force_reset_envs_before_iteration=True,
                evaluation_num_env_runners=1,
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=False
            ).api_stack(
                enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
            )
        )

        run_config = RunConfig(
            name=self.experiment.name,
            verbose=2,
            storage_path=self.experiment.storage_path,
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=self.experiment.checkpoint_at_end,
                checkpoint_frequency=self.experiment.checkpoint_frequency,
                checkpoint_score_attribute=self.experiment.checkpoint_score_attribute,
                checkpoint_score_order=self.experiment.checkpoint_score_order,
            ),
            stop={"training_iteration": self.experiment.num_of_episodes * self.experiment.num_env_runners},
        )

        return config, run_config

    def _get_tuner_args(self) -> tuple[dict[str, Any], RunConfig]:
        config, run_config = self.create_env_with_config()
        param_space = config.to_dict()
        param_space.update(self.experiment.get_param_space())
        return param_space, run_config


    def fit(
            self,
    ) -> AlgorithmConfig:
        param_space, run_config = self._get_tuner_args()
        tune.Tuner(
            "DQN",
            run_config=run_config,
            param_space=param_space,
            tune_config=self.experiment.tune_config,
        ).fit()
        return AlgorithmConfig.from_dict(param_space)

    def fit_from_tuner(self) -> AlgorithmConfig:
        param_space, _ = self._get_tuner_args()
        tuner = tune.Tuner.restore(self.experiment.restore_path, self.experiment.algo_name)
        self.create_env()
        tuner.fit()
        return AlgorithmConfig.from_dict(param_space)
