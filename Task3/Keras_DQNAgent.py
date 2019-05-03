# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:08:21 2019

@author:
    
Keras based DQN
Source:
https://keon.io/deep-q-learning/
"""

from collections import deque
import numpy as np
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# Deep Q-learning Agent
class Keras_DQNAgent:
    
    def __init__(self, state_size, action_size, model_filename = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        #if replay_counter reaches replay_frequency, do a replay
        self.replay_frequency = 50
        self.replay_counter = 0
        
        
        if model_filename == None:
            self.model = self._build_model()
        else:
            self.model = load_model(model_filename)
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state,[1,self.state_size])
        next_state = np.reshape(next_state,[1,self.state_size])
        self.memory.append((state, action, reward, next_state, done))
        
        #Check if its time to do a replay
        self.replay_counter += 1
        if self.replay_counter == self.replay_frequency:
            self.replay()
            self.replay_counter = 0
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_filename):
        self.model.save(model_filename)

