from pathlib import Path
from typing import Mapping

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig, APPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env import EnvContext
from ray.train import RunConfig, CheckpointConfig
from ray.tune import register_env
from sumo_rl import SumoEnvironment

from typings.algorithms import ALGORITHM_NAMES
from typings.experiments import Experiment
from wrappers.gym import CustomObservationWrapper

CONFIG_MAPPER: Mapping[ALGORITHM_NAMES, type[AlgorithmConfig]] = {
    "DQN": DQNConfig,
    "PPO": PPOConfig,
    "APPO": APPOConfig,
    "DDQN": DQNConfig,
}


def create_env(
    env_name: str, rou_path: Path, net_path: Path, out_csv_path: Path
) -> None:
    def env_creator(env_config: EnvContext):
        env = SumoEnvironment(
            net_file=str(net_path),
            route_file=str(rou_path),
            out_csv_name=str(out_csv_path),
            single_agent=True,
            use_gui=False,
            num_seconds=3000,
            yellow_time=4,
            min_green=5,
            max_green=60,
            reward_fn="pressure",
            add_system_info=True,
        )
        return CustomObservationWrapper(env)

    register_env(env_name, env_creator)


def create_env_with_config(experiment: Experiment) -> tuple[AlgorithmConfig, RunConfig]:
    create_env(
        env_name=experiment.experiment_type,
        net_path=Path(experiment.net_file),
        rou_path=Path(experiment.rou_file),
        out_csv_path=Path(experiment.out_csv_path),
    )

    config = (
        CONFIG_MAPPER[experiment.algo_name]()
        .environment(env=experiment.experiment_type, disable_env_checking=True)
        .rollouts(num_env_runners=1, rollout_fragment_length=128)
        .training(**experiment.config.model_dump())
        .debugging(log_level=experiment.log_level)
        .framework(framework=experiment.framework)
        .resources(num_gpus=experiment.num_gpus)
        .reporting(min_sample_timesteps_per_iteration=1000)
    )

    config.api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )

    run_config = RunConfig(
        verbose=2,
        storage_path=experiment.storage_path,
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=experiment.checkpoint_at_end,
            checkpoint_frequency=experiment.checkpoint_freq,
            checkpoint_score_attribute=experiment.checkpoint_score_attribute,
            checkpoint_score_order=experiment.checkpoint_score_order,
        ),
        stop={"training_iteration": experiment.stop_after_iteration},
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
