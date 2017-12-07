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
import simple_tag_utilities

# TODO:
# save weights
# record loss
# OpenAI record video

def play(episodes, is_render, is_testing, checkpoint_interval, \
        weights_filename_prefix, csv_filename_prefix, batch_size):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["epoch"]
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

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
        if not is_testing:
            losses = []
            size = memories[0].pointer
            batch = random.sample(range(size), size) if size < batch_size else random.sample(
                range(size), batch_size)

            for i in range(env.n):
                if done[i]:
                    rewards[i] *= 100

                memories[i].remember(states, actions, rewards[i],
                                     states_next, done[i])

                if memories[i].pointer > batch_size * 10:
                    s, a, r, sn, _ = memories[i].sample(batch, env.n)
                    r = np.reshape(r, (batch_size, 1))
                    critics[i].learn(s, a, r, sn)
                    actors[i].learn(s[i])
                    # TODO: losses.append(history.history["loss"][0])
                    losses.append(0)
                else:
                    losses.append(-1)

            states = states_next

            # collect statistics and print rewards. NOTE: simple tag specific!
            statistic = [episode]
            statistic.extend([rewards[i] for i in range(env.n)])
            statistic.extend([losses[i] for i in range(env.n)])
            for i in range(env.n):
                collide_i = 0
                for j in range(i+1, env.n):
                    is_collide = simple_tag_utilities.is_collision(env.agents[i], \
                                                                   env.agents[j])
                    if is_collide and env.agents[i].adversary is not env.agents[j].adversary:
                        collide_i += 1
                statistic.append(collide_i)
            statistics.add_statistics(statistic)
            if episode % 25 == 0:
                print('Episode: ', episode, ' Rewards: ', rewards)

        # reset states if done
        if any(done):
            states = env.reset()

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(csv_filename_prefix, episode))
            general_utilities.save_dqn_weights(critics,
                                               weights_filename_prefix + "critic_")
            general_utilities.save_dqn_weights(actors,
                                               weights_filename_prefix + "actor_")
            if episode >= checkpoint_interval:
                os.remove("{}_{}.csv".format(csv_filename_prefix, \
                        episode - checkpoint_interval))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--video_dir', default='videos/', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=100000, type=int)
    parser.add_argument('--video_interval', default=1000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-dqn',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

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
        memories.append(Memory(args.memory_size))

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

    general_utilities.load_dqn_weights_if_exist(
        actors, args.experiment_prefix + args.weights_filename_prefix + "actor_", ".h5.index")
    general_utilities.load_dqn_weights_if_exist(
        critics, args.experiment_prefix + args.weights_filename_prefix + "critic_", ".h5.index")

    start_time = time.time()

    # play
    statistics = play(args.episodes, args.render, args.testing,
            args.checkpoint_frequency,
            args.experiment_prefix + args.weights_filename_prefix,
            args.experiment_prefix + args.csv_filename_prefix,
            args.batch_size)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    tf.summary.FileWriter(args.experiment_prefix + args.weights_filename_prefix, session.graph)
    general_utilities.save_dqn_weights(critics,
                                       args.experiment_prefix + args.weights_filename_prefix + "critic_")
    general_utilities.save_dqn_weights(actors,
                                       args.experiment_prefix + args.weights_filename_prefix + "actor_")
    statistics.dump(args.csv_filename_prefix + ".csv")
