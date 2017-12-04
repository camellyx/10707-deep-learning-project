import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from dqn import DQN
from make_env import make_env
from tag_utilities import Tag_Actions
import general_utilities

def main():
    parser = argparse.ArgumentParser()
    # Can also use 'simple_tag'
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--train_episodes', default=1e3, type=int)
    parser.add_argument('--render', default=False,
                        action="store_true")
    parser.add_argument('--benchmark', default=False,
                        action="store_true")
    parser.add_argument('--weights_filename_prefix',
                        default='./save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--testing', default=False,
                        action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--naive', default=False,
                        action="store_true",
                        help="disables all agents except one")
    parser.add_argument('--random_seed', default=2, type=int)
    options = parser.parse_args()
    start_time = time.time()

    np.random.seed(options.random_seed)
    random.seed(options.random_seed)
    env = make_env(options.env, options.benchmark)
    print("action space", env.action_space)
    print("observation space", env.observation_space)
    print("env.n", env.n)
    print("env.world.dim_c", env.world.dim_c)
    dqns = [DQN(env.action_space[agent_i].n, env.observation_space[agent_i].shape[0]) for agent_i in range(env.n)]
    general_utilities.load_dqn_weights_if_exist(dqns, options.weights_filename_prefix)
    statistics_header = ["epoch", "reward_0", "reward_1", "loss_0", "loss_1"]
    statistics = general_utilities.Time_Series_Statistics_Store(statistics_header)
    state = env.reset()
    movement_rate = 0.1
    for step in itertools.count():
        t = (step + 1) * 0.005 if not options.testing else 1000
        if step >= options.train_episodes:
            break
        if options.render:
            env.render()
        agent_actions = []
        actions = []
        for agent_i in range(env.n):
            # Calculate agent policy
            if agent_i != 0:
                a = dqns[agent_i].choose_action(state[agent_i], t) if not options.naive \
                                                                   else Tag_Actions.STOP.value
            else:
                a = dqns[agent_i].choose_action(state[agent_i], t)
            onehot_action = np.zeros(4 + env.world.dim_c)
            onehot_action[a] = 1 * movement_rate
            agent_actions.append(onehot_action)
            actions.append(a)
        state_next, reward, done, info = env.step(agent_actions)
        if step % 25 == 0:
            print("Step {step} with reward {reward}".format(step=step, reward=reward))
        losses = []
        for i in range(env.n):
            dqns[i].remember(state[i], actions[i], reward[i], state_next[i], done[i])
            if step > 500:
                history = dqns[i].learn()
                losses.append(history.history["loss"][0])
            else:
                losses.append(-1)

        statistics.add_statistics([step, reward[0], reward[1], losses[0], losses[1]])
        state = state_next

        if any(done):
            state = env.reset()
            env.render()
    total_time = time.time() - start_time
    print("Finished {} episodes in {} seconds".format(options.train_episodes, total_time))
    general_utilities.save_dqn_weights(dqns, options.weights_filename_prefix)
    statistics.dump("statistics.csv")

if __name__ == '__main__':
    main()
    # code.interact(local=dict(globals(), **locals()))
