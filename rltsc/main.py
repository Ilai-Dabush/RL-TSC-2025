from rltsc.training.trainer import Trainer


def run(experiment_name: str) -> None:
    trainer = Trainer(experiment_name=experiment_name)
    trainer.fit()

def run_from_checkpoint(experiment_name: str) -> None:
    trainer = Trainer(experiment_name=experiment_name)
    trainer.fit_from_tuner()



if __name__ == "__main__":
    # run("DQN")
    run_from_checkpoint("DQN")
