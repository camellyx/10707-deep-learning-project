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

from make_env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--train_episodes', default=1e3, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    options = parser.parse_args()

    env = make_env(options.env, options.benchmark)
    state = env.reset()
    for step in itertools.count():
        if step >= options.train_episodes:
            break
        if options.render:
            env.render()
        action = []
        for i in range(env.n):
            # See policy.py: InteractivePolicy.action()
            action.append(np.random.random(4 + env.world.dim_c))
        next_state, reward, done, info = env.step(action)
        if any(done):
            env.render()
            break


if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
