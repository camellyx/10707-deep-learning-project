# 10707-deep-learning-project
Deep Learning Course Project

## Useful links

* [OpenAI blog on multi-agent RL](https://blog.openai.com/learning-to-cooperate-compete-and-communicate/)

### Traffic Coordination

* [Open Traffic Collection](https://github.com/graphhopper/open-traffic-collection)

### simple_tag state and action

* state (obs) of an agent[i] consists of 16 values: state[0:2] velocity of agent[i], state[2:4] coordinate of agent[i], state[4:8] relative coordinate of 2 landmarks, state[8:14] relative coordinate of 3 other agents, state[14:16] communication channel (the adversary does not have this)
* action[0:7] is one hot encoding of [no_action, right, left, down, up, 2 x communication (the adversary does not have this)]
