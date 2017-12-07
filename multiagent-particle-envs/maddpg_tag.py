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
from functools import reduce

from ddpg import Actor
from maddpg import Critic
from memory import Memory
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from make_env import make_env
import general_utilities

MEMORY_SIZE = 10000
BATCH_SIZE = 64

# TODO:
# save weights
# record loss
# OpenAI record video


def play():
    states = env.reset()
    for episode in range(args.episodes):
        # render
        if args.render:
            env.render()

        # act
        actions = []
        for i in range(env.n):
            action = np.clip(
                actors[i].choose_action(states[i]) + actors_noise[i](), -2, 2)
            actions.append(action)

        # step
        states_next, rewards, done, info = env.step(actions)

        # learn
        if not args.testing:
            losses = []
            size = memories[0].pointer
            batch = random.sample(range(size), size) if size < BATCH_SIZE else random.sample(
                range(size), BATCH_SIZE)

            for i in range(env.n):
                if done[i]:
                    rewards[i] *= 100

                memories[i].remember(states, actions, rewards[i],
                                     states_next, done[i])

                if memories[i].pointer > BATCH_SIZE * 10:
                    s, a, r, sn, _ = memories[i].sample(batch, env.n)
                    r = np.reshape(r, (BATCH_SIZE, 1))
                    critics[i].learn(s, a, r, sn)
                    actors[i].learn(s[i])
                    # TODO: losses.append(history.history["loss"][0])
                    losses.append(0)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=100000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--weights_filename_prefix', default='./save/tag-maddpg/',
                        help="where to store/load network weights")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    args = parser.parse_args()

    # init env
    env = make_env(args.env, args.benchmark)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init actors and critics
    session = tf.Session()

    n_actions = []
    actors = []
    actors_noise = []
    memories = []
    eval_actions = []
    target_actions = []
    state_placeholders = []
    state_next_placeholders = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]
        state = tf.placeholder(tf.float32, shape=[None, state_size])
        state_next = tf.placeholder(tf.float32, shape=[None, state_size])
        speed = 0.9 if env.agents[i].adversary else 1

        actors.append(Actor('actor' + str(i), session, n_action, speed,
                            state, state_next))
        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(n_action)))
        memories.append(Memory(MEMORY_SIZE))

        n_actions.append(n_action)
        eval_actions.append(actors[i].eval_actions)
        target_actions.append(actors[i].target_actions)
        state_placeholders.append(state)
        state_next_placeholders.append(state_next)

    critics = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        reward = tf.placeholder(tf.float32, [None, 1])

        critics.append(Critic('critic' + str(i), session, n_actions,
                              eval_actions, target_actions, state_placeholders,
                              state_next_placeholders, reward))
        actors[i].add_gradients(critics[i].action_gradients[i])

    session.run(tf.global_variables_initializer())

    # init statistics. NOTE: simple tag specific!
    statistics_header = ["epoch", "reward_0", "reward_1", "loss_0", "loss_1"]
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)
    start_time = time.time()

    # play
    play()

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    tf.summary.FileWriter(args.weights_filename_prefix, session.graph)
    statistics.dump("./save/statistics-maddpg.csv")
