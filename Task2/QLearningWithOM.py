# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:46:33 2019

@author:
    
Combined Q-learning with opponent modelling
    
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

class qomPlayer:
    
    def __init__(self,alpha = 0.1, initialBeliefs = [1,1,1]):
        
        #Q function: AxA -> Q value
        #AxA: Action of self and Action of other player
        q_init = 100.0
        self.Q = np.array([[q_init,q_init,q_init],
                           [q_init,q_init,q_init],
                           [q_init,q_init,q_init]])
        self.alpha = alpha
        
        #Opponent Modelling
        self.beliefs = np.array(initialBeliefs)
        if initialBeliefs[0] == initialBeliefs[1] and initialBeliefs[0] == initialBeliefs[2]:
            self.expected = random.randint(0,2)
        else:
            self.expected = initialBeliefs.index(max(initialBeliefs))
        
        #Expected Value for player's actions:
        self.EV = np.zeros(3)
        for i in range(0,3):
            self.EV[i] = sum( np.multiply(self.Q[i,:], self.beliefs / float(sum(self.beliefs))) )
        
        #count random actions
        self.counter = 0
        
        #Track total earned reward
        self.total_reward = 0
        
    def play(self,k):
        #Find all actions with maximum EV. Select a random action of these.
        possible_actions = np.argwhere(self.EV == np.amax(self.EV)).ravel()
        r = random.randint(0,possible_actions.shape[0]-1)
        action = possible_actions[r]
        
        # Exploration vs Exploitation
        #action = self.explore(action,k)
        
        return action
        
    # Use Boltzmann Exploration to possibly alter the action
    # Does not work yet! Problem because of Q values
    def explore(self,action,k):
        prob_action = k**self.EV[action] / (k**self.EV[0] + k**self.EV[1] + k**self.EV[2])
        print("prob_action: " + str(prob_action))
        new_action = action
        if random.random() > prob_action:
            new_action = math.trunc((3*random.random()))
            self.counter += 1
        
        return new_action
    
    def update(self, myAction, opponentAction, reward):
        #Update reward count
        self.total_reward += reward
        print(str(self.total_reward))
        #Update Q
        self.Q[myAction,opponentAction] = (1 - self.alpha)*self.Q[myAction,opponentAction] + self.alpha*reward
        
        #Update beliefs
        self.beliefs[opponentAction] += 1
        
        #Update EV
        self.EV[myAction] = sum( np.multiply(self.Q[myAction,:], self.beliefs / float(sum(self.beliefs))) )
        

class RPSgame:
    # Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
    #         Rock Paper Scissor
    # Rock
    # Paper
    # Scissor
    reward_bimatrix = np.array([[[0,0], [-1,1], [1,-1]],
                                [[1,-1], [0,0], [-1,1]],
                                [[-1,1], [1,-1], [0,0]]])
    reward_matrix = reward_bimatrix[:,:,0]
    
    def __init__(self, player1 = qomPlayer(), player2 = qomPlayer()):
        self.p1 = player1
        self.p2 = player2
        self.results = np.zeros([3,3])
        
    def play(self, episodes = 1000):
        for ep in range(episodes):
            
            k = ep + 1.0
            
            play_p1 = self.p1.play(k)
            play_p2 = self.p2.play(k)
            payoff_p1, payoff_p2 = self.reward_bimatrix[play_p1,play_p2]
            self.p1.update(play_p1,play_p2,payoff_p1)
            self.p2.update(play_p2,play_p1,payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results


game = RPSgame()
results = game.play()
print(results)
print("Row player: " + str(results.sum(axis=1) / sum(sum(results))))
print("Column player: " + str(results.sum(axis=0) / sum(sum(results))))
