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

import maddpg
from dqn import DQN
from memory import Memory
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from make_env import make_env
import general_utilities
import simple_tag_utilities


def play(episodes, is_render, is_testing, checkpoint_interval,
         weights_filename_prefix, csv_filename_prefix, batch_size):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["eps_greedy_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0

        while True:
            steps += 1

            # render
            if args.render:
                env.render()

            # act
            actions = []
            for i in range(env.n):
                if i < h:
                    action = dqns[i].choose_action(states[i])
                    speed = 0.9 if env.agents[i].adversary else 1

                    onehot_action = np.zeros(n_actions[i])
                    onehot_action[action] = speed
                    actions.append(onehot_action)
                else:
                    action = np.clip(
                        actors[i].choose_action(states[i]) + actors_noise[i](), -2, 2)
                    actions.append(action)

            # step
            states_next, rewards, done, info = env.step(actions)

            # learn
            if not args.testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)

                for i in range(env.n):
                    if done[i]:
                        rewards[i] -= 50

                    if i < h:
                        memories[i].remember(states[i], np.argmax(actions[i]),
                                             rewards[i], states_next[i], done[i])
                    else:
                        memories[i].remember(states, actions, rewards[i],
                                             states_next, done[i])

                    if i < h:
                        if memories[i].pointer > batch_size * 10:
                            s, a, r, sn, done = memories[i].sample(batch)
                            history = dqns[i].learn(s, a, r, sn, done)
                            episode_losses[i] += history.history["loss"][0]
                        else:
                            episode_losses[i] = -1
                    else:
                        if memories[i].pointer > batch_size * 10:
                            s, a, r, sn, _ = memories[i].sample(batch, env.n)
                            r = np.reshape(r, (batch_size, 1))
                            loss = critics[i].learn(s, a, r, sn)
                            actors[i].learn(actors, s)
                            episode_losses[i] += loss
                        else:
                            episode_losses[i] = -1

            states = states_next
            episode_rewards += rewards
            collision_count += np.array(
                simple_tag_utilities.count_agent_collisions(env))

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n)])
                statistic.extend([episode_losses[i] for i in range(env.n)])
                statistic.extend([*[dqns[i].eps_greedy for i in range(h)],
                                  *[-1 for i in range(h, env.n-h+1)]])
                statistic.extend(collision_count.tolist())
                statistics.add_statistics(statistic)
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                break

        if episode % checkpoint_interval == 0:
            if i < h:
                statistics.dump("{}_{}.csv".format(csv_filename_prefix,
                                                   episode))
                general_utilities.save_dqn_weights(dqns,
                                                   "{}_{}_".format(weights_filename_prefix, episode))
                if episode >= checkpoint_interval:
                    os.remove("{}_{}.csv".format(csv_filename_prefix,
                                                 episode - checkpoint_interval))
            else:
                statistics.dump("{}_{}.csv".format(
                    csv_filename_prefix, episode))
                if not os.path.exists(weights_filename_prefix):
                    os.makedirs(weights_filename_prefix)
                save_path = saver.save(session, os.path.join(
                    weights_filename_prefix, "models"), global_step=episode)
                print("saving model to {}".format(save_path))
                if episode >= checkpoint_interval:
                    os.remove("{}_{}.csv".format(csv_filename_prefix,
                                                 episode - checkpoint_interval))

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
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon_greedy', nargs='+', type=float,
                        help="Epsilon greedy parameter for each agent")
    parser.add_argument('--ou_mus', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_sigma', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    parser.add_argument('--ou_theta', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_dt', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise dt for each agent")
    parser.add_argument('--ou_x0', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise x0 for each agent")
    parser.add_argument('--load_weights_from_file', default='',
                        help="where to load network weights")
    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")
    # init env
    env = make_env(args.env, args.benchmark)

    if args.epsilon_greedy is not None:
        if len(args.epsilon_greedy) == env.n:
            epsilon_greedy = args.epsilon_greedy
        else:
            raise ValueError("Must have enough epsilon_greedy for all agents")
    else:
        epsilon_greedy = [0.5 for i in range(env.n)]

    # Extract ou initialization values
    if args.ou_mus is not None:
        if len(args.ou_mus) == sum([env.action_space[i].n for i in range(env.n)]):
            ou_mus = []
            prev_idx = 0
            for space in env.action_space:
                ou_mus.append(
                    np.array(args.ou_mus[prev_idx:prev_idx + space.n]))
                prev_idx = space.n
            print("Using ou_mus: {}".format(ou_mus))
        else:
            raise ValueError(
                "Must have enough ou_mus for all actions for all agents")
    else:
        ou_mus = [np.zeros(env.action_space[i].n) for i in range(env.n)]

    if args.ou_sigma is not None:
        if len(args.ou_sigma) == env.n:
            ou_sigma = args.ou_sigma
        else:
            raise ValueError("Must have enough ou_sigma for all agents")
    else:
        ou_sigma = [0.3 for i in range(env.n)]

    if args.ou_theta is not None:
        if len(args.ou_theta) == env.n:
            ou_theta = args.ou_theta
        else:
            raise ValueError("Must have enough ou_theta for all agents")
    else:
        ou_theta = [0.15 for i in range(env.n)]

    if args.ou_dt is not None:
        if len(args.ou_dt) == env.n:
            ou_dt = args.ou_dt
        else:
            raise ValueError("Must have enough ou_dt for all agents")
    else:
        ou_dt = [1e-2 for i in range(env.n)]

    if args.ou_x0 is not None:
        if len(args.ou_x0) == env.n:
            ou_x0 = args.ou_x0
        else:
            raise ValueError("Must have enough ou_x0 for all agents")
    else:
        ou_x0 = [None for i in range(env.n)]

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # split agents
    # agent_id < h is dqn
    # agent_id >= h is maddpg
    h=1

    # init actors and critics
    session = tf.Session()

    # General agent data
    n_actions = [env.action_space[i].n for i in range(env.n)]
    state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
    print("n_actions", n_actions)
    print("state_sizes", state_sizes)
    print("network types", ["dqn" if i < h else "maddpg" for i in range(env.n)])

    # init DQNs
    dqn_memories = [Memory(args.memory_size) for i in range(h)]
    dqns = [DQN(n_actions[i], state_sizes[i], eps_greedy=epsilon_greedy[i])
            for i in range(h)]

    # load DQN
    general_utilities.load_dqn_weights_if_exist(
        dqns, args.experiment_prefix + args.weights_filename_prefix)

    maddpg_actors = []
    maddpg_actors_noise = []
    maddpg_memories = []

    # TODO How to integrate with q learning?
    maddpg_eval_actions = [] # [dqn.eval_network for dqn in dqns]
    maddpg_target_actions = [] #[dqn.target_network for dqn in dqns]
    maddpg_state_placeholders = [tf.placeholder(tf.float32, shape=[None, state_sizes[i]]) for i in range(h)]
    maddpg_state_next_placeholders = [tf.placeholder(tf.float32, shape=[None, state_sizes[i]]) for i in range(h)]

    maddpg_critics = []
    for i in range(h, env.n-h+1):
        state = tf.placeholder(tf.float32, shape=[None, state_sizes[i]])
        state_next = tf.placeholder(tf.float32, shape=[None, state_sizes[i]])
        speed = 0.9 if env.agents[i].adversary else 1

        # Actor
        maddpg_actors.append(maddpg.Actor('actor' + str(i), session, n_actions[i], speed,
                            state, state_next))
        maddpg_actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=ou_mus[i],
            sigma=ou_sigma[i],
            theta=ou_theta[i],
            dt=ou_dt[i],
            x0=ou_x0[i]))
        maddpg_memories.append(Memory(args.memory_size))

        maddpg_eval_actions.append(maddpg_actors[h-i].eval_actions)
        maddpg_target_actions.append(maddpg_actors[h-i].target_actions)
        maddpg_state_placeholders.append(state)
        maddpg_state_next_placeholders.append(state_next)

    for i in range(h, env.n-h+1):
        # Critic
        reward = tf.placeholder(tf.float32, [None, 1])
        maddpg_critics.append(maddpg.Critic('critic' + str(i), session, n_actions,
                              maddpg_eval_actions, maddpg_target_actions, maddpg_state_placeholders,
                              maddpg_state_next_placeholders, reward))
        maddpg_actors[h-i].add_gradients(maddpg_critics[h-i].action_gradients[h-i])


    actors = [*[None for i in range(h)], *maddpg_actors]
    critics = [*[None for i in range(h)], *maddpg_critics]
    actors_noise = [*[None for i in range(h)], *maddpg_actors_noise]
    memories = [*dqn_memories, *maddpg_memories]
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000000)

    if args.load_weights_from_file != "":
        saver.restore(session, args.load_weights_from_file)
        print("restoring from checkpoint {}".format(
            args.load_weights_from_file))

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
    general_utilities.save_dqn_weights(
        dqns, args.experiment_prefix + args.weights_filename_prefix)
    tf.summary.FileWriter(args.experiment_prefix +
                          args.weights_filename_prefix, session.graph)
    save_path = saver.save(session, os.path.join(
        args.experiment_prefix + args.weights_filename_prefix, "models"), global_step=args.episodes)
    print("saving model to {}".format(save_path))
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
