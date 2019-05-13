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
import os
import csv
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

# Deep Q-learning Agent
class Keras_DQNAgent:
    
    def __init__(self, state_size, action_size, model_name):
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name
        self.model_filename = model_name + ".h5"
        self.loss_per_minibatch_filename = model_name + "_loss_per_minibatch.csv"
        self.loss_total_filename = model_name + "_loss_total.csv"
        
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
        self.episode_loss_total = 0.0
        self.episode_loss_per_minibatch = list()
        
        
        
        if os.path.isfile(self.model_filename) == False:
            self.model = self._build_model()
            with open(self.loss_per_minibatch_filename, 'w',newline='') as wf:
                writer = csv.writer(wf)
                l = ["Loss function for " + self.model_name + " calculated per " + str(self.batch_size) + " samples."]
                writer.writerow(l)
            wf.close()
            with open(self.loss_total_filename, 'w',newline='') as wf:
                writer = csv.writer(wf)
                l = ["Loss function for " + self.model_name + " calculated per episode."]
                writer.writerow(l)
            wf.close()
            
        else:
            self.model = load_model(self.model_filename)
        
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
        state = np.reshape(state,[1,self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        minibatch_loss = 0.0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            prediction = target_f[0][action]
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
            minibatch_loss += (target - prediction) ** 2
            #self.episode_steps += 1
            
        minibatch_loss /= float(self.batch_size)   
        self.episode_loss_per_minibatch.append([minibatch_loss])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
  
    def save_model(self):
        self.model.save(self.model_filename)

    def save_loss(self):
        if (len(self.episode_loss_per_minibatch) > 0):
            # Save loss per minibatch
            with open(self.loss_per_minibatch_filename, 'a',newline='') as wf:
                writer = csv.writer(wf)
                writer.writerows(self.episode_loss_per_minibatch)
            wf.close()   
            
            # Save loss per episode
            temp = np.array(self.episode_loss_per_minibatch).flatten()
            self.episode_loss_total = sum(temp) / float(temp.size)
            with open(self.loss_total_filename, 'a',newline='') as wf:
                writer = csv.writer(wf)
                writer.writerow([self.episode_loss_total])
            wf.close()           
    
    def end_episode(self):
        self.save_model()
        self.save_loss()
    