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
from keras.models import load_model

from custom_DRL_models import default_model, cnn_model_v1, cnn_model_v2

"""
# Double Deep Q-learning Agent
This model uses two networks. 
While training (replay), the weights of the Primary are updated using Q-values of the Secondary.
For predicting actions, the Primary is used.

"""
class Keras_DDQNAgent:
    
    def __init__(self, state_size, action_size, model_name):
        self.state_shape = (15,15)
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name
        self.model1_filename = model_name + "_PRIMARY.h5"
        #self.model2_filename = model_name + "_SECONDARY.h5"
        self.update_model_counter = 0
        self.loss_per_minibatch_filename = model_name + "_loss_per_minibatch.csv"
        self.loss_total_filename = model_name + "_loss_total.csv"
        
        self.memory = deque(maxlen=500)
        self.batch_size = 500
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        #if replay_counter reaches replay_frequency, do a replay
        self.replay_frequency = 500
        self.replay_counter = 0
        self.episode_loss_total = 0.0
        self.episode_loss_per_minibatch = list()
        
        
        
        
        if os.path.isfile(self.model1_filename) == False:
            # Init models AND files to store progress of loss function corresponding to this model
            self.model1 = self._build_model()
            self.model2 = self._build_model()
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
            self.model1 = load_model(self.model1_filename)
            self.model2 = load_model(self.model1_filename)
        
    def _build_model(self):
        return default_model(input_shape=(15,15,1), action_size=self.action_size)
        #return cnn_model_v2(input_shape=(15,15,1), action_size = self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        #state = np.reshape(state,[1,self.state_size])
        #next_state = np.reshape(next_state,[1,self.state_size])
        state = self.reshape_state(state)
        next_state = self.reshape_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
        
        #Check if its time to do a replay
        self.replay_counter += 1
        if self.replay_counter == self.replay_frequency:
            self.replay()
            self.replay_counter = 0
        
        
    def act(self, state):
        state = self.reshape_state(state)
        act_values = self.model1.predict(state)
        return act_values[0]  # returns q-values Q(state,.)
    
    def replay(self):
        for _ in range(0,2):
            minibatch = random.sample(self.memory, self.batch_size)
            minibatch_loss = 0.0
            for state, action, reward, next_state, done in minibatch:
                target_1 = reward
                if not done:
                    next_argmax_1 = np.argmax(self.model1.predict(next_state)[0])
                    next_Q_2 = self.model2.predict(next_state)[0][next_argmax_1]
                    target_1 = reward + self.gamma * next_Q_2
                target_f = self.model1.predict(state)
                prediction = target_f[0][action]
                target_f[0][action] = target_1
                self.model1.fit(state, target_f, epochs=1, verbose=0)
                
                minibatch_loss += (target_1 - prediction) ** 2
                self.update_model_counter += 1
            
            # update loss
            minibatch_loss /= float(self.batch_size)   
            self.episode_loss_per_minibatch.append([minibatch_loss])
        
        # update model
        if self.update_model_counter > 1000:
            self.update_models()
            self.update_model_counter = 0
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reshape_state(self, state):
        return np.reshape(state,(1,15,15,1))
        
    def update_models(self):
        self.model1.save(self.model1_filename)
        self.model2 = load_model(self.model1_filename)
    
    
    def save_model(self):
        self.model1.save(self.model1_filename)
        #self.model2.save(self.model2_filename)

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
        self.update_models()
        self.save_model()
        self.save_loss()
        print("\n\nDDQN Agent finished.\n")
    