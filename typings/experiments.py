from functools import cached_property
from typing import (
    Annotated,
    Optional,
    List,
    TypeAlias,
    Literal,
    Generic,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, PositiveInt, PositiveFloat
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from typings.algorithms import ALGORITHM_NAMES

T = TypeVar("T")


class ExperimentBaseConfig(BaseModel):
    train_batch_size: PositiveInt
    lr: PositiveFloat
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
    dueling: bool
    double_q: bool
    hiddens: List[PositiveInt]
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
    args: List[T]


class ParamSpaceConfig(BaseModel):
    lr: ParamConfig[PositiveFloat]
    gamma: ParamConfig[PositiveFloat]


class DQNParamSpaceConfig(ParamSpaceConfig):
    algo_name: Literal["DQN", "DDQN"]
    target_network_update_freq: ParamConfig[List[PositiveInt]]
    hiddens: ParamConfig[List[List[PositiveInt]]]
    n_step: ParamConfig[List[PositiveInt]]
    adam_epsilon: ParamConfig[PositiveFloat]
    train_batch_size: ParamConfig[List[PositiveInt]]


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
    experiment_type: str
    algo_name: ALGORITHM_NAMES
    num_gpus: Annotated[int, Field(default=1)]
    log_level: Annotated[str, Field(default="ERROR")]
    checkpoint_at_end: Annotated[bool, Field(default=True)]
    checkpoint_frequency: Annotated[int, Field(default=10)]
    stop_after_iteration: Annotated[int, Field(default=1000)]
    framework: Annotated[str, Field(default="torch")]
    checkpoint_score_attribute: Annotated[
        str, Field(default="env_runners/episode_reward_mean")
    ]
    checkpoint_score_order: Annotated[str, Field(default="max")]
    num_of_episodes: PositiveInt
    checkpoint_freq: PositiveInt
    num_env_runners: PositiveInt
    net_file: Annotated[str, Field(default="nets/sumo/sumo_net.net.xml")]
    rou_file: Annotated[str, Field(default="nets/sumo/sumo_net.rou.xml")]
    out_csv_path: Annotated[str, Field(default="out/sumo/sumo_net.csv")]
    config: Union[DQNExperimentConfig, PPOExperimentConfig, APPOExperimentConfig] = (
        Field(discriminator="algo_name")
    )
    param_space: Union[
        DQNParamSpaceConfig, APPOParamSpaceConfig, PPOParamSpaceConfig
    ] = Field(discriminator="algo_name")

    @property
    def storage_path(self) -> str:
        return f"/ilai/results/{self.experiment_type}"

    @property
    def checkpoints_path(self) -> str:
        return f"{self.storage_path}/{self.algo_name}"

    @cached_property
    def tune_config(self) -> tune.TuneConfig:
        scheduler = ASHAScheduler(
            metric=self.checkpoint_score_attribute,
            mode=self.checkpoint_score_order,
            grace_period=3,
            reduction_factor=2,
            # single num episodes >= grace_period
            max_t=500,
        )

        return tune.TuneConfig(scheduler=scheduler, num_samples=3)
