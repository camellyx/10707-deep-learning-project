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

def play(episodes, is_render, is_testing, checkpoint_interval, \
        weights_filename_prefix, csv_filename_prefix):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["epoch"]
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    states = env.reset()
    speed = 0.1

    for episode in range(episodes):
        # render
        if is_render:
            env.render()

        # act
        actions = []
        actions_onehot = []
        for i in range(env.n):
            action = dqns[i].choose_action(states[i])

            onehot_action = np.zeros(n_actions[i])
            onehot_action[action] = 1 * speed
            actions_onehot.append(onehot_action)
            actions.append(action)

        # step
        states_next, rewards, done, info = env.step(actions_onehot)

        # learn
        if not is_testing:
            losses = []
            size = memories[0].pointer
            batch = random.sample(range(size), size) if size < BATCH_SIZE else random.sample(
                range(size), BATCH_SIZE)

            for i in range(env.n):
                if done[i]:
                    rewards[i] *= 100

                memories[i].remember(states[i], actions[i],
                                     rewards[i], states_next[i], done[i])

                if memories[i].pointer > BATCH_SIZE * 10:
                    history = dqns[i].learn(*memories[i].sample(batch))
                    losses.append(history.history["loss"][0])
                else:
                    losses.append(-1)

            states = states_next

            # collect statistics and print rewards. NOTE: simple tag specific!
            statistics.add_statistics([episode, rewards[0], rewards[1],
                                       losses[0], losses[1]])
            print('Episode: ', episode, ' Rewards: ', rewards)

        # reset states if done
        if any(done):
            states = env.reset()

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(csv_filename_prefix, episode))
            general_utilities.save_dqn_weights(dqns, \
                    "{}_{}_".format(weights_filename_prefix, episode))

    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=500000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-dqn',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
            args.experiment_prefix + "/save/run_parameters.json")

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
    memories = [Memory(MEMORY_SIZE, 2 * state_sizes[i] + 3)
                for i in range(env.n)]
    dqns = [DQN(n_actions[i], state_sizes[i]) for i in range(env.n)]

    general_utilities.load_dqn_weights_if_exist(
        dqns, args.experiment_prefix + args.weights_filename_prefix)

    start_time = time.time()

    # play
    statistics = play(args.episodes, args.render, args.testing,
            args.checkpoint_frequency,
            args.experiment_prefix + args.weights_filename_prefix,
            args.experiment_prefix + args.csv_filename_prefix)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    general_utilities.save_dqn_weights(dqns, args.experiment_prefix + args.weights_filename_prefix)
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
