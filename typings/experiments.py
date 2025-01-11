from typing import Annotated, Optional, List

from pydantic import BaseModel, Field, PositiveInt, PositiveFloat


class ExperimentBaseConfig(BaseModel):
    train_batch_size: PositiveInt
    lr: PositiveFloat
    gamma: PositiveFloat
    grad_clip: Optional[PositiveFloat]
    store_buffer_in_checkpoints: Annotated[bool, Field(default=False)]
    adam_epsilon: PositiveFloat
    v_max: Annotated[float, Field(default=10.0)]
    v_min: Annotated[float, Field(default=-10.0)]
    num_atoms: PositiveInt
    noisy: Annotated[bool, Field(default=False)]
    sigma0: PositiveFloat


class DQNExperimentConfig(ExperimentBaseConfig):
    target_network_update_freq: PositiveInt
    dueling: bool
    double_q: bool
    hiddens: List[PositiveInt]
    n_step: PositiveInt
    training_intensity: Optional[PositiveFloat]


class Experiment(BaseModel):
    experiment_type: str
    algo_name: str
    log_level: Annotated[str, Field(default="ERROR")]
    num_of_episodes: PositiveInt
    checkpoint_freq: PositiveInt
    num_env_runners: PositiveInt
    config: ExperimentBaseConfig