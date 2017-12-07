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

from ddpg import Actor, Critic
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
    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.array([0.0] * env.n)
        episode_rewards = np.array([0.0] * env.n)
        steps = 0

        while True:
            steps += 1

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
            episode_rewards += rewards

            # learn
            if not args.testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < BATCH_SIZE else random.sample(
                    range(size), BATCH_SIZE)

                for i in range(env.n):
                    if done[i]:
                        rewards[i] *= 100

                    memories[i].remember(states[i], actions[i],
                                         rewards[i], states_next[i], done[i])

                    if memories[i].pointer > BATCH_SIZE * 10:
                        s, a, r, sn, _ = memories[i].sample(batch)
                        r = np.reshape(r, (BATCH_SIZE, 1))
                        critics[i].learn(s, a, r, sn)
                        actors[i].learn(s)
                        # TODO: episode_losses[i] += history.history["loss"][0]
                    else:
                        episode_losses[i] = -1

                states = states_next

            # reset states if done
            if any(done):
                # collect statistics and print rewards
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps
                statistics.add_statistics([episode,
                                           episode_rewards[0], episode_rewards[1],
                                           episode_losses[0], episode_losses[1]])
                print('Episode: ', episode,
                      ' Steps: ', steps,
                      ' Rewards: ', episode_rewards,
                      ' Losses: ', episode_losses)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=100000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--weights_filename_prefix', default='./save/tag-ddpg/',
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

    actors = []
    critics = []
    actors_noise = []
    memories = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]
        state = tf.placeholder(tf.float32, shape=[None, state_size])
        reward = tf.placeholder(tf.float32, [None, 1])
        state_next = tf.placeholder(tf.float32, shape=[None, state_size])
        speed = 0.9 if env.agents[i].adversary else 1

        actors.append(Actor('actor' + str(i), session, n_action, speed,
                            state, state_next))
        critics.append(Critic('critic' + str(i), session, n_action,
                              actors[i].eval_actions, actors[i].target_actions,
                              state, state_next, reward))
        actors[i].add_gradients(critics[i].action_gradients)
        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(n_action)))
        memories.append(Memory(MEMORY_SIZE))

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
    statistics.dump("./save/statistics-ddpg.csv")
