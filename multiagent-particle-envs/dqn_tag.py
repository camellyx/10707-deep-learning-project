import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
from collections import namedtuple
import code
import errno
from dqn import DQN

from make_env import make_env

def ensure_directory_exists(base_directory):
    try:
        os.makedirs(base_directory)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise ex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--train_episodes', default=1e5, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    parser.add_argument('--weights_filename_prefix', default='./save/tag-dqn')
    options = parser.parse_args()

    env = make_env(options.env, options.benchmark)
    print("action space", env.action_space)
    print("observation space", env.observation_space)
    print("env.n", env.n)
    print("env.world.dim_c", env.world.dim_c)
    dqns = [DQN(env.action_space[agent_i].n, env.observation_space[agent_i].shape[0]) for agent_i in range(env.n)]
    for i, dqn in enumerate(dqns):
        # TODO should not work if only some weights available?
        dqn_filename = options.weights_filename_prefix + str(i) + ".h5"
        if os.path.isfile(dqn_filename):
            print("Found old weights to use for {}".format(i))
            dqn.load(dqn_filename)
    state = env.reset()
    movement_rate = 0.01
    for step in itertools.count():
        t = (step + 1) * 0.005
        if step >= options.train_episodes:
            break
        if options.render:
            env.render()
        agent_actions = []
        actions = []
        for agent_i in range(env.n):
            # Calculate agent policy
            a = dqns[agent_i].choose_action(state[agent_i], t)
            onehot_action = np.zeros(4 + env.world.dim_c)
            onehot_action[a] = 1 * movement_rate
            agent_actions.append(onehot_action)
            actions.append(a)
        #print("agent_actions", agent_actions)
        state, reward, done, info = env.step(agent_actions)
        print("reward", reward)

        for i in range(env.n):
            dqns[i].remember(state[i], actions[i], reward[i], state_next[i], done)
            if step > 500:
                dqns[i].learn()

        state = state_next
        
        if any(done):
            env.render()
            break
    ensure_directory_exists(os.path.splitext(options.weights_filename_prefix)[0])
    for i, dqn in enumerate(dqns):
        dqn_filename = options.weights_filename_prefix + str(i) + ".h5"
        dqn.save(dqn_filename)

if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
