from __future__ import division
import gym
import argparse

import numpy as np
# import tensorflow as tf
import itertools
import time
import os
import pickle as pk
from collections import namedtuple
import code

from make_env import make_env

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class Agent(object):

    def __init__(self):
        self.training = True
        self.step = 1


def fill_memory(options, env, memories):
    for i in range(20):
        state = env.reset()
        for step in range(50):
            if options.render:
                env.render()
            actions = []
            onehot_actions = []
            for i in range(env.n):
                if i == env.n - 1:
                    role = 1
                else:
                    role = 0
                action = np.random.randint(env.action_space[i].n)
                actions.append(action)
                onehot_action = np.zeros(env.action_space[i].n)
                onehot_action[action] = options.movement_rate
                onehot_actions.append(onehot_action)

            next_state, reward, done, info = env.step(onehot_actions)
            reward = np.clip(reward, -1., 1.)

            for i in range(env.n):
                if i == env.n - 1:
                    role = 1
                else:
                    role = 0
                memories[role].append(state[i], actions[i], reward[i], done[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str)
    parser.add_argument('--folder', default='', type=str)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--movement_rate', default=1., type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_episodes', default=1e6, type=int)
    parser.add_argument('--linear_size', default=50, type=int)
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--window_length', default=1, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    options = parser.parse_args()
    if options.folder == "":
        options.folder = options.env
    if not os.path.isdir(options.folder):
        os.makedirs(options.folder)

    env = make_env(options.env, options.benchmark)
    np.random.seed(123)
    env.seed(123)

    # TODO: This is the code for separate DQN for each agent, need to tweak
    # Keras DQN
    n_actions = [env.action_space[0].n, env.action_space[-1].n]
    n_states = [env.observation_space[0].shape[0],
                env.observation_space[-1].shape[0]]
    filename = options.folder + "/agent.pk"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            agent = pk.load(f)
    else:
        agent = Agent()

    models = []
    memories = []
    policies = []
    for n in range(len(n_actions)):
        filename = options.folder + "/model_" + str(n) + ".h5"
        if os.path.isfile(filename):
            model = load_model(filename)
        else:
            model = Sequential()
            model.add(Dense(options.linear_size, activation='relu',
                            input_shape=(n_states[n],)))
            model.add(Dense(options.linear_size, activation='relu'))
            model.add(Dense(n_actions[n], activation='softmax'))
            model.compile(optimizer=Adam(
                lr=options.learning_rate), loss='mse', metrics=['mae'])
            print(model.summary())
        models.append(model)
        memories.append(SequentialMemory(
            limit=options.memory_size, window_length=options.window_length))
        policies.append(LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                             value_max=1., value_min=.1,
                                             value_test=.05,
                                             nb_steps=1000000))
        policies[-1]._set_agent(agent)

    fill_memory(options, env, memories)

    epoch = 1
    last_epoch_step = 0
    state = env.reset()
    for step in itertools.count(agent.step):
        agent.step = step
        if epoch >= options.train_episodes:
            break
        if options.render:
            env.render()
        actions = []
        onehot_actions = []
        for i in range(env.n):
            if i == env.n - 1:
                role = 1
            else:
                role = 0
            # action.append(np.random.random(4 + env.world.dim_c))
            q_value = models[role].predict(state[i].reshape(1, -1))
            action = policies[role].select_action(q_values=q_value[0])
            actions.append(action)
            onehot_action = np.zeros(4 + env.world.dim_c)
            onehot_action[action] = options.movement_rate
            onehot_actions.append(onehot_action)

        next_state, reward, done, info = env.step(onehot_actions)
        done = [any(reward[:3])] * 4
        reward = np.clip(reward, -1., 1.)

        for i in range(env.n):
            if i == env.n - 1:
                role = 1
            else:
                role = 0
            memories[role].append(state[i], actions[i], reward[i], done[i])

        losses = []
        my_history = []
        for role in range(len(n_actions)):
            experiences = memories[role].sample(options.batch_size)

            # Start by extracting the necessary parameters (we use a
            # vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0[0])
                state1_batch.append(e.state1[0])
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = np.array(state0_batch)
            state1_batch = np.array(state1_batch)
            reward_batch = np.array(reward_batch)
            terminal1_batch = np.array(terminal1_batch)

            target_q_values = models[role].predict(
                state1_batch, batch_size=options.batch_size)
            q_batch = np.max(target_q_values, axis=1).flatten()

            targets = np.zeros((options.batch_size, n_actions[role]))

            discounted_reward_batch = options.gamma * q_batch * terminal1_batch
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, R, action) in enumerate(zip(targets, Rs, action_batch)):
                target[action] = R
            history = models[role].fit(state0_batch, targets,
                                       batch_size=options.batch_size, verbose=0)
            losses.append(history.history['loss'][-1])
            my_history.append(history.history)

        if step % 100 == 0:
            print(epoch, step, losses)
        if step % 1000 == 0:
            for n in range(len(n_actions)):
                filename = options.folder + "/model_" + str(n) + ".h5"
                models[n].save(filename)
            filename = options.folder + "/agent.pk"
            with open(filename, 'wb') as f:
                pk.dump(agent, f)
            filename = options.folder + "/history.pk"
            with open(filename, 'wb') as f:
                pk.dump(my_history, f)

        if any(done) or step - last_epoch_step > 100:
            if any(done):
                print("done! in", step - last_epoch_step, "steps")
            state = env.reset()
            epoch += 1
            last_epoch_step = step


if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
