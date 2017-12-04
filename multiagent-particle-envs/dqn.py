import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

# TODO: prioritized replay buffer

class DQN:
    def __init__(self, n_actions, state_size, epsilon=0.5, gamma=0.9, batch_size=64, memory_size=2000, replace_target_steps=1000):
        self.n_actions = n_actions
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.learning_step = 0
        self.eval_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_weights()
        self.epsilon=epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_steps = replace_target_steps

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
        if p < (1 - self.epsilon / t):
            action_probs = self.eval_network.predict(state[np.newaxis, :])
            return np.argmax(action_probs[0])
        else:
            return random.randrange(self.n_actions)

    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))

    def learn(self):
        if self.learning_step % self.replace_target_steps == 0:
            self.update_target_weights()

        minibatch = random.sample(self.memory, self.batch_size)
        minibatch_state, minibatch_action, minibatch_reward, \
                minibatch_state_next, minibatch_done = map(np.array, zip(*minibatch))

        minibatch_not_done = np.logical_not(minibatch_done)

        minibatch_e = self.eval_network.predict(minibatch_state_next)
        minibatch_t = self.target_network.predict(minibatch_state_next)

        best_action = np.argmax(minibatch_e, axis=1)
        discounted_reward = self.gamma * minibatch_t[np.arange(minibatch_t.shape[0]), best_action]

        minibatch_target = self.eval_network.predict(minibatch_state)
        minibatch_target[np.arange(minibatch_target.shape[0]), minibatch_action] = minibatch_reward
        minibatch_target[minibatch_not_done, minibatch_action[minibatch_not_done]] += discounted_reward[minibatch_not_done]

        history = self.eval_network.fit(minibatch_state, minibatch_target, epochs=1, verbose=0)
        self.learning_step += 1
        return history

    def load(self, name):
        self.eval_network.load_weights(name)
        self.target_network.load_weights(name)

    def save(self, name):
        self.eval_network.save_weights(name)
        self.target_network.save_weights(name)

