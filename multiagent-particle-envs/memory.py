import numpy as np
import random
from collections import deque


class Memory:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
        self.data.append(experience)
        if self.pointer < len(self.data):
            self.pointer += 1

    def sample(self, batch):
        states = np.array([self.data[i][0] for i in batch])
        actions = np.array([self.data[i][1] for i in batch])
        rewards = np.array([self.data[i][2] for i in batch])
        states_next = np.array([self.data[i][3] for i in batch])
        dones = np.array([self.data[i][4] for i in batch])

        return states, actions, rewards, states_next, dones
