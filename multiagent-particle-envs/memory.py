import numpy as np
import random
from collections import deque


class Memory:
    def __init__(self, capacity, dimension):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
        self.data.append(experience)
        self.pointer += 1

    def sample(self, n):
        batch = []
        if self.pointer < n:
            batch = random.sample(self.data, self.pointer)
        else:
            batch = random.sample(self.data, n)

        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        states_next = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        return states, actions, rewards, states_next, dones
