# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:24 2019

@author: serru
"""

import random
import numpy as np

class QLearningPlayer:
    
    def __init__(self,alpha = 0.1):
        
        #Q function: AxA -> Q value
        #AxA: Action of self and Action of other player
        q_init = 0.0
        self.Q = np.array([q_init,q_init,q_init])
        #Learning rate alpha
        self.alpha = alpha
        #discout factor gamma
        self.gamma = 0.9
        
        #Policy and corresponding max Q-value for player's actions:
        self.Pi = random.randrange(0,3)
        
        #keep track of probabilities of a certain action
        self.probs_timeline = np.array([[0,0,0]])
        
        #count number of actions not corresponding to Pi due to exploration.
        self.counter = 0
        
        #Track total earned reward
        self.total_reward = 0
        self.total_reward_timeline = list()
        
    def play(self,T):
            
        #Use Boltzmann Exploration to calculate probabilities of choosing a certain action
        # Nominator
        p = np.e**(self.Q / float(T))
        p = p.flatten()
        #Devide with denominator
        denom = sum(p)
        self.probs = p / denom
        self.probs_timeline = np.append(self.probs_timeline,[np.copy(self.probs)],axis=0)
        
        #Use cumulative sum + random number to choose an action
        probs_cs = np.cumsum(self.probs)
        r = random.random()
        for new_action in range(0,3):
            if (r < probs_cs[new_action]): 
                if new_action != self.Pi:
                    self.counter += 1
                return new_action

    
    def update(self, myAction, opponentAction, reward):
        #Note: In basic Q-learning, the opponentAction cannot be used!!!
        
        #Update reward count
        self.total_reward += reward
        self.total_reward_timeline.append(self.total_reward)
        
        #Update Q
        self.Q[myAction] = (1 - self.alpha)*self.Q[myAction] + self.alpha*(reward + self.gamma*np.max(self.Q))
        
        #Update Policy
        self.Pi = np.argmax(self.Q)