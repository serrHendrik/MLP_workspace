# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:24 2019

@author: serru
"""

import random
import numpy as np

class QFPPlayer:
    
    def __init__(self,alpha = 0.1, initialBeliefs = [1,1,1]):
        
        #Q function: AxA -> Q value
        #AxA: Action of self and Action of other player
        q_init = 0.0
        self.Q = np.array([[q_init,q_init,q_init],
                           [q_init,q_init,q_init],
                           [q_init,q_init,q_init]])
        self.alpha = alpha
        self.gamma = 0.1
        
        #Opponent Modelling
        self.beliefs = np.array(initialBeliefs)
        
        #beliefs_timeline is used for visualisation purposes
        self.beliefs_timeline = np.array([self.beliefs / float(sum(self.beliefs))])
        
        #Expected Value for player's actions:
        self.EV = np.zeros(3)
        for i in range(0,3):
            self.EV[i] = sum( np.multiply(self.Q[i,:], self.beliefs / float(sum(self.beliefs))) )
        
        #count number of actions not corresponding to largest EV due to exploration.
        self.counter = 0
        
        #Track total earned reward
        self.total_reward = 0
        self.total_reward_timeline = list()
        
    def play(self,T):
        
        #Calculate action with largest EV (only used for counter statistics)
        action_largestEV = np.argmax(self.EV)
        
        #Use Boltzmann Exploration to calculate probabilities of choosing a certain action
        # Nominator
        probs = np.e**(self.EV / float(T))
        probs = probs.flatten()
        #print("probs nominator: " + str(probs))
        #Devide with denominator
        denom = sum(probs)
        probs = probs / denom
        
        #Use cumulative sum + random number to choose an action
        probs_cs = np.cumsum(probs)
        r = random.random()
        for new_action in range(0,3):
            if (r < probs_cs[new_action]): 
                if new_action != action_largestEV:
                    self.counter += 1
                return new_action

    
    def update(self, myAction, opponentAction, reward):
        #Update reward count
        self.total_reward += reward
        self.total_reward_timeline.append(self.total_reward)
        
        #Update Q
        self.Q[myAction,opponentAction] = (1 - self.alpha)*self.Q[myAction,opponentAction] + self.alpha*(reward + self.gamma*np.max(self.EV))
        
        #Update beliefs
        self.beliefs[opponentAction] += 1
        self.beliefs_timeline = np.append(self.beliefs_timeline,[np.copy(self.beliefs) / float(sum(self.beliefs))],axis=0)
        
        #Update EV
        #self.EV[myAction] = sum( np.multiply(self.Q[myAction,:], self.beliefs / float(sum(self.beliefs))) )
        for i in range(0,3):
            self.EV[i] = sum( np.multiply(self.Q[i,:], self.beliefs / float(sum(self.beliefs))) )