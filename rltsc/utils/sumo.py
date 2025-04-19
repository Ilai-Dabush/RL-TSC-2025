from pathlib import Path
from typing import Mapping

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig, APPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env import EnvContext
from ray.train import RunConfig, CheckpointConfig
from ray.tune import register_env
from sumo_rl import SumoEnvironment

from rltsc.callbacks.resources import ResourcesCallback
from rltsc.typings.algorithms import ALGORITHM_NAMES
from rltsc.typings.experiments import Experiment
from rltsc.wrappers.gym import CustomObservationWrapper

CONFIG_MAPPER: Mapping[ALGORITHM_NAMES, type[AlgorithmConfig]] = {
    "DQN": DQNConfig,
    "PPO": PPOConfig,
    "APPO": APPOConfig,
    "DDQN": DQNConfig,
}


def create_env(
        experiment: Experiment
) -> None:
    def env_creator(env_config: EnvContext):
        env = SumoEnvironment(
            net_file=experiment.net_file,
            route_file=experiment.rou_file,
            out_csv_name=experiment.out_csv_path,
            single_agent=True,
            use_gui=False,
            # num_seconds=20000,
            yellow_time=experiment.min_yellow_time,
            min_green=experiment.min_green_time,
            reward_fn=experiment.reward_fn,
            add_system_info=True,
        )
        return CustomObservationWrapper(env)

    register_env(experiment.experiment_type, env_creator)


def create_env_with_config(experiment: Experiment) -> tuple[AlgorithmConfig, RunConfig]:
    create_env(
        experiment=experiment,
    )

    config = (
        CONFIG_MAPPER[experiment.algo_name]()
        .environment(env=experiment.experiment_type, disable_env_checking=True)
        .callbacks(ResourcesCallback)
        .env_runners(num_env_runners=experiment.num_env_runners, rollout_fragment_length=100, num_envs_per_env_runner=1,create_env_on_local_worker=True)  #
        .learners(num_learners=2, num_gpus_per_learner=0.5, num_cpus_per_learner=1)
        .training(**experiment.config.model_dump(exclude={"algo_name"}),
                  replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer', "capacity": 50000,
                                        "alpha": 0.6,
                                        # Beta parameter for sampling from prioritized replay buffer.
                                        "beta": 0.4})
        .debugging(log_level=experiment.log_level)
        .framework(framework=experiment.framework)
        .resources(num_gpus=experiment.num_gpus)
        # .reporting()
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_num_env_runners=1,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False
        )
    )

    config.api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )

    run_config = RunConfig(
        verbose=2,
        storage_path=experiment.storage_path,
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=experiment.checkpoint_at_end,
            checkpoint_frequency=experiment.checkpoint_frequency,
            checkpoint_score_attribute=experiment.checkpoint_score_attribute,
            checkpoint_score_order=experiment.checkpoint_score_order,
        ),
        stop={"training_iteration": experiment.num_of_episodes * experiment.num_env_runners, "timesteps_total": 100_000},
    )

    return config, run_config


def fit(
        experiment: Experiment,
) -> None:
    config, run_config = create_env_with_config(experiment)
    tune.Tuner(
        "DQN",
        run_config=run_config,
        param_space=config.to_dict(),
        tune_config=experiment.tune_config,
    ).fit()
