'''
This Agent uses Q-Learning to play games.

The Q-function is represented as an ANN. 

Future:
    https://arxiv.org/pdf/1507.00814.pdf
    https://arxiv.org/pdf/1506.02142.pdf

Inspiration:
    https://arxiv.org/abs/1312.5602.pdf
'''
import gym
import os
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


class Simulation:
    def __init__(self, epsilon=1, epsilon_min=0, gamma=0.99, 
            replay_batch_size=20, replay_ary_len=100):
        self.env = gym.make('CartPole-v1')

        # gym parameters
        self.input_size = self.env.observation_space.shape[0]
        self.actions = range(0, self.env.action_space.n)
        self.output_size = len(self.actions)

        # algorithm parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.replay_ary_len = replay_ary_len
        self.replay_batch_size = replay_batch_size

        # simulation parameters
        self.render = False

        # Q-function, represented as an ANN
        self.model = Sequential([
            Dense(8, input_dim=self.input_size),
            Activation('relu'),
            Dense(16),
            Activation('relu'),
            Dense(self.output_size),
        ])
        self.model.compile('adam', loss='mse')
        self.replay = []

    # epsilon-greedy action selection. epsilon is decayed with each epoch.
    def _choose_action(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q = self.model.predict(state.reshape(1, self.input_size))
            action = np.argmax(q[0])

        return action

    # https://arxiv.org/pdf/1312.5602.pdf (Algorithm 1)
    def _generate_minibatch(self):
        batch_size = min(len(self.replay), self.replay_batch_size)
        batch = random.sample(self.replay, batch_size)
        X_train = np.zeros((batch_size, self.input_size))
        Y_train = np.zeros((batch_size, self.output_size))

        for i in range(batch_size):
            state, reward, action, done, old_state = batch[i]

            old_q = self.model.predict(old_state.reshape(1, self.input_size))[0]
            new_q = self.model.predict(state.reshape(1, self.input_size))[0]
            Y_target = np.copy(old_q)

            if done:
                Y_target[action] = reward
            else:
                Y_target[action] = reward + (self.gamma * np.max(new_q))

            # predict the future reward of a given state
            X_train[i] = old_state
            Y_train[i] = Y_target

        return X_train, Y_train

    def _train_iter(self):
        state_tensor = self.env.reset()
        reward_epoch = 0
        loss = 0
        step = 0

        positions = []
        state_history = []
        while True:  # each iteration is a single frame/step of game
            step += 1
            if self.render:
                self.env.render()

            # choose an action, and measure the reward
            action = self._choose_action(state_tensor)
            old_state = state_tensor
            state_tensor, reward, done, info = self.env.step(action)
            reward_epoch += reward

            # save replay history
            self.replay.append([state_tensor, reward, action, done, old_state])
            if len(self.replay) > self.replay_ary_len:
                self.replay.pop(random.randint(0, self.replay_ary_len))

            # train on a random sample of prior runs
            X_train, Y_train = self._generate_minibatch()
            loss += self.model.train_on_batch(X_train, Y_train)

            if done or step > 200:
                break

        return reward_epoch, loss

    def train(self, epoch_n): 
        for epoch in range(epoch_n):
            reward, loss = self._train_iter()
            print('epoch: {} reward: {} epsilon: {:.4f} loss: {:.4f}'.format(
                epoch, reward, self.epsilon, loss))

            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon = 1 - epoch / epoch_n


if __name__=='__main__':
    s = Simulation()
    s.train(epoch_n=500)
