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
from dqn import DQN

from make_env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--train_episodes', default=1e5, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    options = parser.parse_args()

    env = make_env(options.env, options.benchmark)
    print("action space", env.action_space)
    print("observation space", env.observation_space)
    print("env.n", env.n)
    print("env.world.dim_c", env.world.dim_c)
    dqns = [DQN(env.action_space[agent_i].n, env.observation_space[agent_i].shape[0]) for agent_i in range(env.n)]
    state = env.reset()
    movement_rate = 0.01
    for step in itertools.count():
        t = (step + 1) * 0.005
        if step >= options.train_episodes:
            break
        if options.render:
            env.render()
        agent_actions = []
        for agent_i in range(env.n):
            # Calculate agent policy
            a = dqns[agent_i].choose_action(state[agent_i], t)
            onehot_action = np.zeros(4 + env.world.dim_c)
            onehot_action[a] = 1 * movement_rate
            agent_actions.append(onehot_action)
        #print("agent_actions", agent_actions)
        state, reward, done, info = env.step(agent_actions)
        print("reward", reward)
        if any(done):
            env.render()
            break


if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
