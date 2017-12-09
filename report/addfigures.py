#!/usr/bin/env python3
import os


def addModel(green, red, figure, model):
    print("  \\begin{subfigure}[h]{\\figscale\linewidth}")
    print(
        "    \includegraphics[trim=10 10 10 10,clip,width=\linewidth]")
    filename = os.path.join(
        "../results", "{}_{}vs{}".format(model, green, red))
    filename = os.path.join(filename, figure + ".png")
    print("    {" + filename + "}")
    if figure == "steps":
        caption = "Number of steps per episode over training episode for {}.".format(
            model.upper())
    elif figure == "reward":
        caption = "Cumulative reward per episode over training episode for {}.".format(
            model.upper())
    elif figure == "collisions":
        caption = "Number of collisions per episode over training episode for {}.".format(
            model.upper())
    elif figure == "loss":
        caption = "Average loss per episode over training episode for {}.".format(
            model.upper())
    print("    \caption{" + caption + "}")
    print(
        "    \label{fig:" + "{}-{}vs{}-{}".format(model, green, red, figure) + "}")
    print("  \end{subfigure}")


def addFigure(green, red, figure):
    # print("")
    # print("\subsubsection{" + model.upper() + "}")
    # print("\label{sec:experiment:" +
    #       "{}vs{}:".format(green, red) + model + "}")
    # print("")
    for model in models:
        addModel(green, red, figure, model)
        if model != models[-1]:
            print("  ~")
        else:
            print("")


def addScenario(green, red):
    adversary = "one adversary" if green == 1 else "two adversaries"
    agent = "one agent" if red == 1 else "two agents"
    Adversary = "One Adversary" if green == 1 else "Two Adversaries"
    Agent = "One Agent" if red == 1 else "Two Agents"
    print("")
    print(
        "\subsection{" + "{} vs. {} Results".format(Agent, Adversary) + "}")
    print("\label{sec:experiment:" +
          "{}vs{}".format(green, red) + "}")
    print("")
    print("")
    print("\\begin{figure}[h]")
    print("  \centering")
    print("")
    for figure in figures:
        addFigure(green, red, figure)
    print("")
    print("  \caption{" +
          "Results for predator-pray with {} vs. {}".format(agent, adversary) + "}")
    print(
        "  \label{fig:" + "{}vs{}".format(green, red) + "}")
    print("\end{figure}")
    print("\FloatBarrier")
    print("")


if __name__ == '__main__':
    models = ['dqn', 'ddpg', 'maddpg']
    figures = ['steps', 'reward', 'collisions', 'loss']
    addScenario(1, 1)
    addScenario(1, 2)
    addScenario(2, 1)
