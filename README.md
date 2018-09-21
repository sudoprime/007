# 007

An openAI gym agent.

### Instructions

```
virtualenv -p python3.5 env
source env/bin/activate
git clone https://github.com/openai/gym.git
git clone https://github.com/openai/universe.git
pip install -e gym
pip install keras tensorflow-gpu # or just tensorflow
pip install -e universe

python run.py
```

### About

This is the start of an openai gym agent that will hopefully be used to learn
how to play arbitrary video games, learning off past experience in other games.

It's based on the atari deep learning nature article, but I left off the
convolutional net since we get a more concise state-representation from gym.
This will presumably change as I work on this agent.

It uses a neural network to represent the Q-function in the Q-learning paradigm.
The neural network get's fed the current state and predicts the future payout of
each possible action. We train the network with replays of our training to speed
up learning.

This initial cut uses epsilon-decreasing action-selection, but convergence is
proving difficult on the MountainCar problem. We will need to more intelligently
search the state-space in order to overcome that problem
