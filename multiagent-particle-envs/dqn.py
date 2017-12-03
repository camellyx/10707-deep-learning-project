import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

EPSILON = 0.5
GAMMA = 0.9
BATCH_SIZE = 32
MEMORY_SIZE = 2000
REPLACE_TARGET_STEPS = 200

# TODO: prioritized replay buffer


class DQN:
    def __init__(self, n_actions, state_size):
        self.n_actions = n_actions
        self.state_size = state_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.learning_step = 0
        self.eval_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_weights()

    def huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def build_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=Adam())

        return model

    def update_target_weights(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def choose_action(self, state, t):
        p = np.random.random()
        if p < (1 - EPSILON / t):
            action_probs = self.eval_network.predict(state[np.newaxis, :])
            return np.argmax(action_probs[0])
        else:
            return random.randrange(self.n_actions)

    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))

    def learn(self):
        if self.learning_step % REPLACE_TARGET_STEPS == 0:
            self.update_target_weights()

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, done in minibatch:
            target = self.eval_network.predict(state[np.newaxis, :])
            if done:
                target[0][action] = reward
            else:
                e = self.eval_network.predict(state_next[np.newaxis, :])[0]
                t = self.target_network.predict(state_next[np.newaxis, :])[0]
                target[0][action] = reward + GAMMA * t[np.argmax(e)]

            self.eval_network.fit(
                state[np.newaxis, :], target, epochs=1, verbose=0)

        self.learning_step += 1
