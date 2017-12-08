#!/usr/bin/env python3
import os


def addFigure(model, green, red, figure):
    print("")
    print("\\begin{figure}[h]")
    print("  \centering")
    print(
        "  \includegraphics[trim=10 10 10 10,clip,width=\\figscale\linewidth]")
    filename = os.path.join(
        "../results", "{}_{}vs{}".format(model, green, red))
    filename = os.path.join(filename, figure + ".png")
    print("  {" + filename + "}")
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
    print("  \caption{" + caption + "}")
    print("  \label{fig:" + "{}-{}vs{}".format(model, green, red) + "}")
    print("\end{figure}")
    print("\FloatBarrier")
    print("")


def addScenario(model, green, red):
    adversary = "One" if green == 1 else "Two"
    agent = "One" if red == 1 else "Two"
    print("")
    print(
        "\subsubsection{" + "{} Agents vs. {} Adversary Results".format(agent, adversary) + "}")
    print("\label{sec:experiment:" +
          "{}:{}vs{}".format(model, green, red) + "}")
    print("")
    for figure in figures:
        addFigure(model, green, red, figure)


def addModel(model):
    print("")
    print("\subsection{" + model.upper() + "}")
    print("\label{sec:experiment:" + model + "}")
    print("")
    addScenario(model, 1, 1)
    addScenario(model, 1, 2)
    addScenario(model, 2, 1)


if __name__ == '__main__':
    models = ['dqn', 'ddpg', 'maddpg']
    figures = ['steps', 'reward', 'collisions', 'loss']
    for model in models:
        addModel(model)
