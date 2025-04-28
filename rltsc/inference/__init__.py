import os

import cloudpickle
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID, \
    COMPONENT_ENV_RUNNER, COMPONENT_MODULE_TO_ENV_CONNECTOR, Columns, COMPONENT_ENV_TO_MODULE_CONNECTOR
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env import EnvContext
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.tune import register_env
from sumo_rl import SumoEnvironment

from rltsc.utils.sumo import bootstrap
from rltsc.wrappers.gym import CustomObservationWrapper


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

    def run_env(self, experiment_type: str, net_file: str, route_file: str, exp_path: str, checkpoint_path: str) -> None:
        ray.shutdown()
        ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
        bootstrap()
        # self.create_env(experiment_type, net_file, route_file)
        base_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            single_agent=True,
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
        prev_action = None

        while not episode.is_done:
            shared_data = {}
            input_dict = env_to_module(
                episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
                rl_module=rl_module,
                explore=False,
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
                explore=False,
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

if __name__ == "__main__":
    # path = r"D:\experiments_checkpoints\DDQN_SingleAgent\DDQN_50_ep\DQN_DDQN_SingleAgent_ba98b_00000_0_adam_epsilon=0.0000,gamma=0.9606,hiddens=256_256,lr=0.0001,n_step=7,target_network_update_freq=_2025-04-21_21-08-10\checkpoint_000027\algorithm_state.pkl"
    # with open(r"D:\experiments_checkpoints\test.pkl", "rb") as f:
    #     data = cloudpickle.load(f)
    #     print(1)
    # with open(r"D:\experiments_checkpoints\test.pkl", "wb") as f:
    #     cloudpickle.dump(data, f)
    runner = Inference()

    runner.run_env("DDQN_SingleAgent",
            r"C:\Users\ilai\Desktop\RL-TSC-2025\rltsc\routes\intersection.net.xml",
            r"C:\Users\ilai\Desktop\RL-TSC-2025\rltsc\routes\intersection.rou.xml",
            r"D:\experiments_checkpoints\DDQN_SingleAgent\DQN_7_iter_no_gpu\DQN_DDQN_SingleAgent_d8d49_00001_1_adam_epsilon=0.0000,gamma=0.9845,hiddens=128_128,lr=0.0000,n_step=5,target_network_update_freq=_2025-04-27_21-27-48",
            r"D:\experiments_checkpoints\DDQN_SingleAgent\DQN_7_iter_no_gpu\DQN_DDQN_SingleAgent_d8d49_00001_1_adam_epsilon=0.0000,gamma=0.9845,hiddens=128_128,lr=0.0000,n_step=5,target_network_update_freq=_2025-04-27_21-27-48\checkpoint_000006"
            )
