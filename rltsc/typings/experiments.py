from typing import (
    Annotated,
    Optional,
    TypeAlias,
    Literal,
    Generic,
    TypeVar,
    Union, Any, Callable,
)
from uuid import uuid4

import ray
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, validate_call
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from rltsc.rewards.pressure import normalized_pressure
from rltsc.typings.algorithms import ALGORITHM_NAMES
from rltsc.typings.enums import Platforms
from rltsc.utils.os_utils import get_platform

T = TypeVar("T")


class ExperimentBaseConfig(BaseModel):
    train_batch_size: Optional[PositiveInt] = None
    lr: Optional[PositiveFloat] = None
    gamma: PositiveFloat
    grad_clip: Optional[PositiveFloat] = None


class BasePPOExperimentConfig(ExperimentBaseConfig):
    clip_param: PositiveFloat
    vf_loss_coeff: Optional[PositiveFloat] = None
    entropy_coeff: Optional[PositiveFloat] = None
    use_gae: bool
    use_critic: bool
    lambda_: PositiveFloat


class PPOExperimentConfig(BasePPOExperimentConfig):
    algo_name: Literal["PPO"]
    sgd_minibatch_size: PositiveInt
    num_sgd_iter: PositiveInt


class APPOExperimentConfig(BasePPOExperimentConfig):
    algo_name: Literal["APPO"]
    use_kl_loss: bool
    kl_coeff: PositiveFloat
    kl_target: PositiveFloat


class DQNExperimentConfig(ExperimentBaseConfig):
    algo_name: Literal["DDQN", "DQN"]
    target_network_update_freq: PositiveInt
    num_steps_sampled_before_learning_starts: Annotated[PositiveInt, Field(default=1000)]
    dueling: bool
    double_q: bool
    hiddens: list[PositiveInt]
    n_step: PositiveInt
    training_intensity: Optional[PositiveFloat] = None
    store_buffer_in_checkpoints: Annotated[bool, Field(default=False)]
    adam_epsilon: PositiveFloat
    v_max: Annotated[float, Field(default=10.0)]
    v_min: Annotated[float, Field(default=-10.0)]
    num_atoms: PositiveInt
    noisy: Annotated[bool, Field(default=False)]
    sigma0: PositiveFloat


ParamSpaceFunc: TypeAlias = Literal[
    "tune.uniform", "tune.loguniform", "tune.choice", "tune.randint"
]


class ParamConfig(BaseModel, Generic[T]):
    func: ParamSpaceFunc
    args: list[T]


class ParamSpaceConfig(BaseModel):
    lr: ParamConfig[PositiveFloat]
    gamma: ParamConfig[PositiveFloat]


class DQNParamSpaceConfig(ParamSpaceConfig):
    algo_name: Literal["DQN", "DDQN"]
    target_network_update_freq: ParamConfig[list[PositiveInt]]
    hiddens: ParamConfig[list[list[PositiveInt]]]
    n_step: ParamConfig[list[PositiveInt]]
    adam_epsilon: ParamConfig[PositiveFloat]
    train_batch_size: ParamConfig[list[PositiveInt]]


class BasePPoParamSpaceConfig(ParamSpaceConfig):
    clip_param: ParamConfig[PositiveFloat]
    lambda_: ParamConfig[PositiveFloat]
    grad_clip: Optional[ParamConfig[PositiveFloat]] = None
    lr: ParamConfig[PositiveFloat]
    gamma: ParamConfig[PositiveFloat]
    vf_loss_coeff: Optional[ParamConfig[PositiveFloat]]
    entropy_coeff: Optional[ParamConfig[PositiveFloat]]


class PPOParamSpaceConfig(BasePPoParamSpaceConfig):
    algo_name: Literal["PPO"]
    num_sgd_iter: ParamConfig[PositiveInt]
    sgd_minibatch_size: Optional[ParamConfig[PositiveInt]] = None


class APPOParamSpaceConfig(BasePPoParamSpaceConfig):
    algo_name: Literal["APPO"]
    kl_coeff: ParamConfig[PositiveFloat]
    kl_target: ParamConfig[PositiveFloat]


class Experiment(BaseModel):
    name: str
    experiment_type: str
    algo_name: ALGORITHM_NAMES
    log_level: Annotated[str, Field(default="ERROR")]
    checkpoint_at_end: Annotated[PositiveInt, Field(default=True)]
    checkpoint_frequency: Annotated[PositiveInt, Field(default=1)]
    stop_after_iteration: Annotated[PositiveInt, Field(default=1000)]
    framework: Annotated[str, Field(default="torch")]
    checkpoint_score_attribute: Annotated[
        str, Field(default="env_runners/episode_return_mean")
    ]
    checkpoint_score_order: Annotated[str, Field(default="max")]
    num_of_episodes: PositiveInt
    num_env_runners: PositiveInt
    min_yellow_time: Annotated[PositiveInt, Field(default=2)]
    min_green_time: Annotated[PositiveInt, Field(default=5)]
    # Pressure is the total amount of exiting vehicles subtracted by the incoming vehicles in all lanes
    reward_fn: Annotated[Union[str, Callable], Field(default=normalized_pressure)]
    restore_from_checkpoint: Optional[str] = None
    config: Union[DQNExperimentConfig, PPOExperimentConfig, APPOExperimentConfig] = (
        Field(discriminator="algo_name")
    )
    param_space: Union[
        DQNParamSpaceConfig, APPOParamSpaceConfig, PPOParamSpaceConfig
    ] = Field(discriminator="algo_name")

    def _pad_with_colab_path(self, path: str, pad_with_slash: bool = False) -> str:
        return f"{'/content/' if get_platform() == Platforms.LINUX else ''}{'/' if pad_with_slash else ''}{path}"

    def get_param_space(self) -> dict[str, Any]:
        @validate_call
        def convert(p: ParamConfig):
            return eval(p.func)(*p.args)

        return {
            k: convert(v) for k, v in self.param_space.model_dump(exclude={"algo_name"}).items()
        }

    @property
    def restore_path(self) -> str:
        return f"{self.storage_path}/{self.name}"


    @property
    def out_csv_path(self)-> str:
        path = f"experiments/outputs/output_{ray.get_runtime_context().get_actor_name()}_{uuid4()}.csv"
        return self._pad_with_colab_path(path)

    @property
    def storage_path(self) -> str:
        return self._pad_with_colab_path(f"experiments/{self.experiment_type}", True)

    @property
    def checkpoints_path(self) -> str:
        return f"{self.storage_path}/{self.algo_name}"

    @property
    def num_gpus(self) -> int:
        return ray.available_resources()["GPU"] if get_platform() == Platforms.LINUX.value else 0

    @property
    def rou_file(self) -> str:
        return self._pad_with_colab_path("rltsc/routes/intersection.rou.xml")

    @property
    def net_file(self) -> str:
        return self._pad_with_colab_path("rltsc/routes/intersection.net.xml")

    @property
    def tune_config(self) -> tune.TuneConfig:
        scheduler = ASHAScheduler(
            metric=self.checkpoint_score_attribute,
            mode=self.checkpoint_score_order,
            grace_period=5,
            reduction_factor=2,
            # single num episodes >= grace_period
            max_t=500,
        )

        return tune.TuneConfig(scheduler=scheduler, num_samples=3, max_concurrent_trials=1, time_budget_s=7200)
