from functools import partial
from typing import Mapping, Any

import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig, APPOConfig, Algorithm
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env import EnvContext
from ray.train import RunConfig, CheckpointConfig
from ray.tune import register_env, ResultGrid
from sumo_rl import SumoEnvironment

from rltsc.callbacks.debug import DebugCallback
from rltsc.callbacks.resources import ResourcesCallback
from rltsc.config import read_config, get_experiment_path_by_name
from rltsc.observation.wrappers.gym import CustomObservationWrapper
from rltsc.rewards.pressure import pressure_clip
from rltsc.typings.algorithms import ALGORITHM_NAMES
from rltsc.typings.experiments import Experiment
from rltsc.utils.sumo import bootstrap

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
        ray.init(num_gpus=self.experiment.num_gpus, ignore_reinit_error=True)

    def create_env(
            self,
    ) -> None:
        pressure_clip_fn = partial(pressure_clip, self.experiment.pressure_clip_hp)

        def env_creator(env_config: EnvContext):
            env = SumoEnvironment(
                net_file=self.experiment.net_file,
                route_file=self.experiment.rou_file,
                out_csv_name=self.experiment.out_csv_path,
                single_agent=True,
                use_gui=False,
                yellow_time=self.experiment.min_yellow_time,
                min_green=self.experiment.min_green_time,
                reward_fn=pressure_clip_fn,
                add_system_info=True,
                observation_class=self.experiment.observation_class
            )
            return CustomObservationWrapper(env)

        register_env(self.experiment.experiment_type, env_creator)

    def create_env_with_config(self) -> tuple[AlgorithmConfig, RunConfig]:
        self.create_env()

        training_args = {
            "replay_buffer_config": {'type': 'PrioritizedEpisodeReplayBuffer',
                                     "capacity": 50000,
                                     "alpha": 0.6,
                                     # Beta parameter for sampling from prioritized replay buffer.
                                     "beta": 0.4,
                                     # "storage_unit": StorageUnit.SEQUENCES,
                                     },
            **self.experiment.config.model_dump(exclude={"algo_name", "override_num_gpus"})
        }

        config = (
            CONFIG_MAPPER[self.experiment.algo_name]()
            .environment(self.experiment.experiment_type, env_config={"horizon": 10_000})
            .callbacks(ResourcesCallback)
            .callbacks(DebugCallback)
            .env_runners(num_env_runners=self.experiment.num_env_runners,
                         exploration_config={
                             "type": "EpsilonGreedy",
                             **self.experiment.exploration_config.model_dump()
                         })
            # .learners(num_learners=2, num_gpus_per_learner=0.5, num_cpus_per_learner=1)
            .training(**training_args)
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

        config["torch_skip_nan_gradients"] = True
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
            stop={"training_iteration": self.experiment.num_iterations},
        )

        return config, run_config

    def _get_tuner_args(self) -> tuple[AlgorithmConfig, dict[str, Any], RunConfig]:
        config, run_config = self.create_env_with_config()
        param_space = config.to_dict()
        param_space.update(self.experiment.get_param_space())
        return config, param_space, run_config

    def _trainable(self, config: dict[str, Any]):
        algo_config, _ = self.create_env_with_config()
        algo: Algorithm = algo_config.build()

        for i in range(config.get("train_iters", 10)):
            result = algo.train()
            tune.report(**result)

            # Optional: manual checkpointing
            if i % 1 == 0:
                checkpoint_dir = f"checkpoint_{i}"
                algo.save(checkpoint_dir)

        algo.stop()

    def fit(
            self,
    ) -> tuple[ResultGrid, AlgorithmConfig]:
        config, param_space, run_config = self._get_tuner_args()
        trainable_with_resources = tune.with_resources(self._trainable, {"cpu": 1})
        results = tune.Tuner(
            self.experiment.algo_name,
            run_config=run_config,
            param_space=param_space,
            tune_config=self.experiment.tune_config,
        ).fit()
        return results, AlgorithmConfig.from_dict(param_space)

    def fit_from_tuner(self) -> tuple[ResultGrid, AlgorithmConfig]:
        _, param_space, _ = self._get_tuner_args()
        tuner = tune.Tuner.restore(self.experiment.restore_path, self.experiment.algo_name)
        self.create_env()
        results = tuner.fit()
        return results, AlgorithmConfig.from_dict(param_space)
