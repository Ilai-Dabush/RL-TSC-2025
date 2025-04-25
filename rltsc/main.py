from ray.rllib.algorithms import AlgorithmConfig

from rltsc.training.trainer import Trainer


def run(experiment_name: str) -> AlgorithmConfig:
    trainer = Trainer(experiment_name=experiment_name)
    return trainer.fit()

def run_from_checkpoint(experiment_name: str) -> AlgorithmConfig:
    trainer = Trainer(experiment_name=experiment_name)
    return trainer.fit_from_tuner()



if __name__ == "__main__":
    run("DQN")
    # run_from_checkpoint("DQN")
