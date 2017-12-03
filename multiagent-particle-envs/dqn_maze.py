from dqn import DQN
from maze_env import Maze

MAX_EPISODES = 500


def train():
    t = 1.0
    step = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        t += 0.005
        print("EPISODE: ", episode)

        while True:
            env.render()
            action = dqn.choose_action(state, t)
            state_next, reward, done = env.step(action)

            dqn.remember(state, action, reward, state_next, done)
            if step > 200 and step % 5 == 0:
                dqn.learn()

            state = state_next
            step += 1

            if done:
                break

    env.destroy()


if __name__ == "__main__":
    env = Maze()
    dqn = DQN(env.n_actions, 2)

    env.after(100, train)
    env.mainloop()
