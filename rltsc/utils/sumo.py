from pathlib import Path
from typing import Mapping

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig, APPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env import EnvContext
from ray.train import RunConfig, CheckpointConfig
from ray.tune import register_env
from sumo_rl import SumoEnvironment

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
        env_name: str, rou_path: Path, net_path: Path, out_csv_path: Path
) -> None:
    def env_creator(env_config: EnvContext):
        env = SumoEnvironment(
            net_file=str(net_path),
            route_file=str(rou_path),
            out_csv_name=str(out_csv_path),
            single_agent=True,
            use_gui=False,
            num_seconds=500,
            yellow_time=4,
            min_green=5,
            max_green=10,
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
        .env_runners(num_env_runners=experiment.num_env_runners, rollout_fragment_length=50)  #  create_env_on_local_worker=True,
        .learners(num_learners=2, num_gpus_per_learner=0.5, num_cpus_per_learner=1)
        .training(**experiment.config.model_dump(exclude={"algo_name"}),
                  replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer', "capacity": 50000,
                                        "alpha": 0.6,
                                        # Beta parameter for sampling from prioritized replay buffer.
                                        "beta": 0.4})
        .debugging(log_level=experiment.log_level)
        .framework(framework=experiment.framework)
        .resources(num_gpus=experiment.num_gpus)
        .reporting(min_sample_timesteps_per_iteration=3000)
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
        stop={"training_iteration": experiment.num_of_episodes * experiment.num_env_runners},
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
