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
from memory import Memory
from make_env import make_env
import general_utilities

MEMORY_SIZE = 10000
BATCH_SIZE = 64


def play():
    states = env.reset()
    speed = 0.1

    for episode in range(args.episodes):
        # render
        if args.render:
            env.render()

        # act
        actions = []
        actions_onehot = []
        for i in range(env.n):
            action = dqns[idx_mapping[i]].choose_action(states[i])

            onehot_action = np.zeros(n_actions[i])
            onehot_action[action] = 1 * speed
            actions_onehot.append(onehot_action)
            actions.append(action)

        # step
        states_next, rewards, done, info = env.step(actions_onehot)

        # learn
        if not args.testing:
            losses = []

            for i in range(env.n):
                if done[i]:
                    rewards[i] *= 100

                memories[idx_mapping[i]].remember(states[i], actions[i],
                                     rewards[i], states_next[i], done[i])

                size = memories[idx_mapping[i]].pointer
                batch = random.sample(range(min(size, BATCH_SIZE)), min(size, BATCH_SIZE))
                if memories[idx_mapping[i]].pointer > BATCH_SIZE * 10:
                    history = dqns[idx_mapping[i]].learn(*memories[idx_mapping[i]].sample(batch))
                    losses.append(history.history["loss"][0])
                else:
                    losses.append(-1)

            states = states_next

            # collect statistics and print rewards. NOTE: simple tag specific!
            statistics.add_statistics([episode] + rewards + losses)
            print('Episode: ', episode, ' Rewards: ', rewards)

        # reset states if done
        if any(done):
            states = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=500000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--weights_filename_prefix', default='./save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--shared_weight', default=False, action="store_true")
    args = parser.parse_args()

    # init env
    env = make_env(args.env, args.benchmark)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init DQNs
    n_actions = [env.action_space[i].n for i in range(env.n)]
    state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
    idx_mapping = {}     # mapping from agent idx to network/memory idx
    for i in range(env.n):
        if env.agents[i].adversary:
            idx_mapping[i] = 0 if args.shared_weight else i
            adversary = i
        else:
            idx_mapping[i] = 1 if args.shared_weight else i
            agent = i

    if args.shared_weight:
        memories = [Memory(MEMORY_SIZE, 2 * state_sizes[i] + 3)
                    for i in [adversary, agent]]
        dqns = [DQN(n_actions[i], state_sizes[i]) for i in [adversary, agent]]
        args.weights_filename_prefix += '_shared'
    else:
        memories = [Memory(MEMORY_SIZE, 2 * state_sizes[i] + 3)
                    for i in range(env.n)]
        dqns = [DQN(n_actions[i], state_sizes[i]) for i in range(env.n)]

    general_utilities.load_dqn_weights_if_exist(
        dqns, args.weights_filename_prefix)

    # init statistics. NOTE: simple tag specific!
    statistics_header = ["epoch"] + ["reward_" + str(i) for i in range(env.n)] + ["loss_" + str(i) for i in range(env.n)]
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)
    start_time = time.time()

    # play
    play()

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    general_utilities.save_dqn_weights(dqns, args.weights_filename_prefix)
    if args.shared_weight:
        statistics.dump("./save/statistics-dqn_shared.csv")
    else:
        statistics.dump("./save/statistics-dqn.csv")
