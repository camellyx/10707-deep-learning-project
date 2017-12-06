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

from monitor import MyMonitor


def play():
    states = env.reset()
    for episode in range(args.episodes):
        # render
        if args.render:
            env.render()

        # act
        actions = []
        for i in range(env.env.n):
            action = np.clip(
                actors[i].choose_action(states[i]) + actors_noise[i](), -2, 2)
            actions.append(action)

        # step
        states_next, rewards, done, info = env.step(actions)

        # learn
        if not args.testing:
            losses = []
            size = memories[0].pointer
            batch = random.sample(range(size), size) if size < args.batch_size else random.sample(
                range(size), args.batch_size)

            for i in range(env.env.n):
                if done[i]:
                    rewards[i] *= 100

                memories[i].remember(states[i], actions[i],
                                     rewards[i], states_next[i], done[i])

                if memories[i].pointer > args.batch_size * 10:
                    s, a, r, sn, _ = memories[i].sample(batch)
                    r = np.reshape(r, (args.batch_size, 1))
                    critics[i].learn(s, a, r, sn)
                    actors[i].learn(s)
                    # TODO: losses.append(history.history["loss"][0])
                    losses.append(0)
                else:
                    losses.append(-1)

            states = states_next

            # collect statistics and print rewards. NOTE: simple tag specific!
            statistic = [episode]
            statistic.extend([rewards[i] for i in range(env.env.n)])
            statistic.extend([losses[i] for i in range(env.env.n)])
            statistics.add_statistics(statistic)
            if episode % 25 == 0:
                print('Episode: ', episode, ' Rewards: ', rewards)

        # reset states if done
        if any(done):
            states = env.reset()

        if episode % args.checkpoint_frequency == 0:
            statistics.dump("{}_{}.csv".format(
                args.csv_filename_prefix, episode))
            general_utilities.save_dqn_weights(critics,
                                               "{}/episode_{}_".format(args.weights_filename_prefix, episode) + "critic_")
            general_utilities.save_dqn_weights(actors,
                                               "{}/episode_{}_".format(args.weights_filename_prefix, episode) + "actor_")
            if episode >= args.checkpoint_frequency:
                os.remove("{}_{}.csv".format(args.csv_filename_prefix,
                                             episode - args.checkpoint_frequency))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=500000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default="save/",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='',
                        help="where to store statistics")
    parser.add_argument('--video_dir', default='videos/ddpg/',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=10000, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--video_interval', default=100, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()
    args.experiment_prefix = os.path.join(args.experiment_prefix, "ddpg")
    args.experiment_prefix = os.path.join(args.experiment_prefix, args.env)
    args.experiment_prefix = os.path.join(
        args.experiment_prefix, 'exp-' + time.strftime("%y-%m-%d-%H-%M"))
    if args.weights_filename_prefix == "":
        args.weights_filename_prefix = os.path.join(
            args.experiment_prefix, "weights/")
    if args.csv_filename_prefix == "":
        args.csv_filename_prefix = os.path.join(
            args.experiment_prefix, "stats/")
    if not os.path.exists(args.weights_filename_prefix):
        os.makedirs(args.weights_filename_prefix)
    if not os.path.exists(args.csv_filename_prefix):
        os.makedirs(args.csv_filename_prefix)

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/run_parameters.json")

    # init env
    env = make_env(args.env, args.benchmark)
    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)
    args.video_dir = os.path.join(
        args.video_dir, 'monitor-' + time.strftime("%y-%m-%d-%H-%M"))
    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)
    env = MyMonitor(env, args.video_dir,
                    # resume=True, write_upon_reset=True,
                    video_callable=lambda episode: (
                        episode + 1) % args.video_interval == 0,
                    force=True)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init actors and critics
    session = tf.Session()

    n_actions = []
    state_sizes = []
    states_placeholder = []
    rewards_placeholder = []
    states_next_placeholder = []
    actors = []
    critics = []
    actors_noise = []
    memories = []
    for i in range(env.env.n):
        n_actions.append(env.action_space[i].n)
        state_sizes.append(env.env.observation_space[i].shape[0])
        states_placeholder.append(tf.placeholder(
            tf.float32, shape=[None, state_sizes[i]]))
        rewards_placeholder.append(tf.placeholder(tf.float32, [None, 1]))
        states_next_placeholder.append(tf.placeholder(tf.float32,
                                                      shape=[None, state_sizes[i]]))
        actors.append(Actor('actor' + str(i), session, n_actions[i], 1,
                            states_placeholder[i], states_next_placeholder[i]))
        critics.append(Critic('critic' + str(i), session, n_actions[i],
                              actors[i].eval_actions, actors[i].target_actions,
                              state_sizes[i], states_placeholder[i],
                              states_next_placeholder[i], rewards_placeholder[i]))
        actors[i].add_gradients(critics[i].action_gradients)
        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(n_actions[i])))
        memories.append(
            Memory(args.memory_size, env.env.n * state_sizes[i] + n_actions[i] + 2))

    session.run(tf.global_variables_initializer())

    # init statistics. NOTE: simple tag specific!
    statistics_header = ["epoch"]
    statistics_header.extend(["reward_{}".format(i) for i in range(env.env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    general_utilities.load_dqn_weights_if_exist(
        actors, args.weights_filename_prefix + "actor_", ".h5.index")
    general_utilities.load_dqn_weights_if_exist(
        critics, args.weights_filename_prefix + "critic_", ".h5.index")

    start_time = time.time()

    # play
    play()

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    tf.summary.FileWriter("options.weights_filename_prefix", session.graph)
    general_utilities.save_dqn_weights(critics,
                                       args.weights_filename_prefix + "critic_")
    general_utilities.save_dqn_weights(actors,
                                       args.weights_filename_prefix + "actor_")
    statistics.dump(args.csv_filename_prefix + ".csv")
