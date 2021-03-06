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
import time
import os
import random
import logging
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Simulation:
    def __init__(self, epsilon=0.05, epsilon_min=0, gamma=0.9, epsilon_target=500,
            replay_batch_size=10000, replay_ary_len=100000):
        self.env = gym.make('Pong-v3')
        logger.info('intitializing simulation.')

        # gym parameters
        self.input_shape = self.env.observation_space.shape
        self.actions = range(0, self.env.action_space.n)
        self.actions_len = len(self.actions)

        # algorithm parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_target = epsilon_target
        self.epsilon_min = epsilon_min
        self.replay_ary_len = replay_ary_len
        self.replay_batch_size = replay_batch_size

        # simulation parameters
        self.render = False

        # Q-function, represented as an ANN
        # https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
        self.model = Sequential([
            Convolution2D(64, 4, 4, border_mode='same', input_shape=self.input_shape),
            Activation('relu'),
            #MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(128, 2, 2, border_mode='same'),
            Activation('relu'),
            #MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            #Dense(64),
            #Activation('relu'),
            Dropout(0.5),
            Dense(self.actions_len),
            Activation('linear'),
        ])
        self.model.compile('adam', loss='mse')
        self.replay = []

    # epsilon-greedy action selection. epsilon is decayed with each epoch.
    def _choose_action(self, state):
        shape_t = list(self.input_shape)
        shape_t.insert(0, 1)

        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q = self.model.predict(state.reshape(shape_t))
            action = np.argmax(q[0])

        return action

    # https://arxiv.org/pdf/1312.5602.pdf (Algorithm 1)
    def _generate_minibatch(self):
        t0 = time.time()
        batch_size = min(len(self.replay), self.replay_batch_size)
        batch = random.sample(self.replay, batch_size)

        inputs = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        targets = np.zeros((batch_size, self.actions_len))

        for i in range(batch_size):
            state, reward, action, done, old_state = batch[i]

            shape_t = list(self.input_shape)
            shape_t.insert(0, 1)

            old_q = self.model.predict(old_state.reshape(shape_t))[0]
            new_q = self.model.predict(state.reshape(shape_t))[0]
            Y_target = np.copy(old_q)

            if done:
                Y_target[action] = reward
            else:
                Y_target[action] = reward + (self.gamma * np.max(new_q))

            # predict the future reward of a given state
            inputs[i] = old_state
            targets[i] = Y_target

        t1 = time.time()
        #print('elapsed time for minibatch: {}s'.format(t1-t0))
        return inputs, targets

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
            inputs, targets = self._generate_minibatch()
            loss += self.model.train_on_batch(inputs, targets)

            if done or step > 10000:
                break

        return reward_epoch, loss

    def train(self, epoch_n): 
        for epoch in range(epoch_n):
            t0 = time.time()
            reward, loss = self._train_iter()
            t1 = time.time()
            logger.info('epoch: {} reward: {} epsilon: {:.4f} loss: {:.4f} time: {}s'.format(
                epoch, reward, self.epsilon, loss, t1-t0))

            # decay epsilon
            #if self.epsilon > self.epsilon_min:
            #    self.epsilon = 1 - epoch / epoch_n


if __name__=='__main__':
    s = Simulation()
    s.train(epoch_n=10000)
