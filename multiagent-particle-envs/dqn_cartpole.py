from dqn import DQN
import gym

MAX_EPISODES = 500


def train():
    t = 1.0
    step = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        t += 0.005
        total_reward = 0
        print("EPISODE: ", episode)

        while True:
            env.render()
            action = dqn.choose_action(state, t)
            state_next, reward, done, _ = env.step(action)

            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = state_next
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / \
                env.theta_threshold_radians - 0.5
            reward = r1 + r2

            dqn.remember(state, action, reward, state_next, done)
            if step > 500:
                dqn.learn()

            state = state_next
            total_reward += reward
            step += 1

            if done:
                print("Reward: ", total_reward)
                break

    env.close()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    print("ACTION SPACE: ", env.action_space)
    print("OBSERVATION SPACE: ", env.observation_space)
    print("OBSERVATION SPACE HIGH: ", env.observation_space.high)
    print("OBSERVATION SPACE LOW: ", env.observation_space.low)

    dqn = DQN(env.action_space.n, env.observation_space.shape[0])
    train()
