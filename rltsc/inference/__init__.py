import os
from functools import partial

import cloudpickle
import ray
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID, \
    COMPONENT_ENV_RUNNER, COMPONENT_MODULE_TO_ENV_CONNECTOR, Columns, COMPONENT_ENV_TO_MODULE_CONNECTOR
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env import EnvContext
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.tune import register_env
from sumo_rl import SumoEnvironment

from rltsc.rewards.pressure import pressure_clip
from rltsc.utils.sumo import bootstrap
from rltsc.observation.wrappers.gym import CustomObservationWrapper


class Inference:

    def create_env(
            self, experiment_type: str, net_file: str, route_file: str,
    ) -> None:

        def env_creator(env_config: EnvContext):
            env = SumoEnvironment(
                net_file=net_file,
                route_file=route_file,
                single_agent=True,
                use_gui=False
            )
            return CustomObservationWrapper(env)

        register_env(experiment_type, env_creator)

    def _edit_gpus_demands(self, checkpoint_path: str) -> None:
        path = os.path.join(checkpoint_path, r"learner_group\learner\rl_module\class_and_ctor_args.pkl")
        data = None
        with open(path, "rb") as rf:
            data = cloudpickle.load(rf)
            # data["num_gpus"] = 0
            data["ctor_args_and_kwargs"][1]["config"]["num_gpus"] = 0

        with open(path, "wb") as f:
            cloudpickle.dump(data, f)

    def run_env(self, net_file: str, route_file: str, checkpoint_path: str) -> None:
        ray.shutdown()
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
        bootstrap()
        # self.create_env(experiment_type, net_file, route_file)
        pressure_clip_fn = partial(pressure_clip, 0.1)
        base_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            num_seconds=3000,
            single_agent=True,
            reward_fn=pressure_clip_fn,
            use_gui=True  # GUI on here, since you're manually running it
        )
        env = CustomObservationWrapper(base_env)
        env_to_module = EnvToModulePipeline.from_checkpoint(
            os.path.join(
                checkpoint_path,
                COMPONENT_ENV_RUNNER,
                COMPONENT_ENV_TO_MODULE_CONNECTOR,
            )
        )
        rl_module = RLModule.from_checkpoint(
            os.path.join(
                checkpoint_path,
                COMPONENT_LEARNER_GROUP,
                COMPONENT_LEARNER,
                COMPONENT_RL_MODULE,
                DEFAULT_MODULE_ID,
            )
        )

        # For the module-to-env pipeline, we will use the convenient config utility.
        module_to_env = ModuleToEnvPipeline.from_checkpoint(
            os.path.join(
                checkpoint_path,
                COMPONENT_ENV_RUNNER,
                COMPONENT_MODULE_TO_ENV_CONNECTOR,
            )
        )
        obs, _ = env.reset()
        episode = SingleAgentEpisode(
            observations=[obs],
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        ray.logger.info("Starting Inference")

        while not episode.is_done:
            shared_data = {}
            input_dict = env_to_module(
                episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
                rl_module=rl_module,
                explore=True,
                shared_data=shared_data,
            )
            # No exploration.
            # if not args.explore_during_inference:
            rl_module_out = rl_module.forward_inference(input_dict)
            # Using exploration.
            # else:
            #     rl_module_out = rl_module.forward_exploration(input_dict)

            to_env = module_to_env(
                batch=rl_module_out,
                episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
                rl_module=rl_module,
                explore=True,
                shared_data=shared_data,
            )
            # Send the computed action to the env. Note that the RLModule and the
            # connector pipelines work on batched data (B=1 in this case), whereas the Env
            # is not vectorized here, so we need to use `action[0]`.
            action = to_env.pop(Columns.ACTIONS)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            # Keep our `SingleAgentEpisode` instance updated at all times.
            episode.add_env_step(
                obs,
                action,
                reward,
                terminated=terminated,
                truncated=truncated,
                # Same here: [0] b/c RLModule output is batched (w/ B=1).
                extra_model_outputs={k: v[0] for k, v in to_env.items()},
            )

        ray.logger.info("Finished Inference")

if __name__ == "__main__":
    runner = Inference()
    runner.run_env(
            r"C:\Users\ilai\Desktop\RL-TSC-2025\rltsc\routes\base-exp\intersection.net.xml",
            r"C:\Users\ilai\Desktop\RL-TSC-2025\rltsc\routes\base-exp\intersection.rou.xml",
            r"D:\experiments_checkpoints\DDQN_SingleAgent\DQN_25_iter_no_gpu_clip_neg0_1_ep_0_2\DQN_DDQN_SingleAgent_5c58f_00003_3_adam_epsilon=0.0000,gamma=0.9785,hiddens=256_256,lr=0.0000,n_step=7,target_network_update_freq=_2025-05-02_20-24-57\checkpoint_000024"
            )
