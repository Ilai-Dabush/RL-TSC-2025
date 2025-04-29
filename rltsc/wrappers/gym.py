import gymnasium as gym
import numpy as np
import torch
import numpy.typing as npt

from rltsc.typings.enums import AlgorithmNames


class Algorithms:

    @staticmethod
    def mashup(obs: npt.NDArray) -> npt.NDArray:
        # Find indices of ones
        ones_indices = np.where(obs == 1)[0]

        # Choose a random percentage between 10% and 30%
        percent_to_flip = np.random.uniform(0.10, 0.30)
        num_to_flip = int(len(ones_indices) * percent_to_flip)

        # Randomly select indices to flip
        indices_to_flip = np.random.choice(ones_indices, size=num_to_flip, replace=False)

        # Flip 1s to 0s at the selected indices
        obs[indices_to_flip] = 0
        return obs

    @staticmethod
    def identity(obs: npt.NDArray) -> npt.NDArray:
        return obs


algorithms_resolver = {
    AlgorithmNames.IDENTITY: Algorithms.identity,
    AlgorithmNames.MAHSUP: Algorithms.mashup,
}


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, algorithm: AlgorithmNames = AlgorithmNames.IDENTITY):
        super(CustomObservationWrapper, self).__init__(env)
        self._algorithm = algorithms_resolver[algorithm]

    def observation(self, obs):
        # Modify the observation here
        modified_obs = self._algorithm(obs)
        return modified_obs
