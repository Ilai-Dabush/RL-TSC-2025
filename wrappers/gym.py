import gymnasium as gym

class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomObservationWrapper, self).__init__(env)

    def observation(self, obs):
        # Modify the observation here
        modified_obs = self.process_observation(obs)
        return modified_obs

    def process_observation(self, obs):
        print(obs)
        # Custom logic to modify observation
        return obs  # Replace with actual transformation