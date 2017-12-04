import numpy as np
from rl.core import Processor


class TagMultiAgentProcessor(Processor):

    def process_observation(self, observation):
        # TODO: we can discretize observation to save memory
        return np.concatenate(observation)

    def process_action(self, action):
        return action

    def process_state_batch(self, batch):
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
