from ray.rllib.algorithms import AlgorithmConfig
from ray.tune import ResultGrid

from rltsc.training.trainer import Trainer


def run(experiment_name: str) -> tuple[ResultGrid, AlgorithmConfig]:
    trainer = Trainer(experiment_name=experiment_name)
    return trainer.fit()

def run_from_checkpoint(experiment_name: str) -> tuple[ResultGrid, AlgorithmConfig]:
    trainer = Trainer(experiment_name=experiment_name)
    return trainer.fit_from_tuner()

def train_from_checkpoint(experiment_name: str) -> None:
    trainer = Trainer(experiment_name=experiment_name)
    trainer.train_from_tuner_checkpoint()



if __name__ == "__main__":
    run("DQN_25_iter_no_gpu_clip_neg0_1_ep_0_2")
    # run_from_checkpoint("DQN")
