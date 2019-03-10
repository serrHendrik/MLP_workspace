# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:49:53 2019

@author: Hendrik Serruys
"""

"""
Replicator Dynamics for the Prisoner's Dilemma Game

Implementation closely follows section 7.7 from Multiagent Systems (Shoham)

"""
import random
import numpy as np

#theta: Action -> proportion of players playing action a at time t
#shape: 2x1
init_fraction = random.random()
theta = np.matrix([[init_fraction] , [1 - init_fraction]])
print 'Initial theta: ' + str(np.transpose(theta))

###
# game_code: Prisoner's Dilemma (0)
#            Matching Pennies (1)
game_code = 0
# Reward / Utility matrix: reward[action_player_0][action_player_1][reward_player_x]
T = 5
R = 3
S = 0
P = 1
reward_pd = np.array([[[R,R],[S,T]] , [[T,S],[P,P]]])
reward_mp = np.array([[[1,-1],[-1,1]] , [[-1,1],[1,-1]]])
reward = reward_pd if game_code == 0 else reward_mp
#epsilon: We generate new generations until the rate of change in the fraction of agents playing action a at time t is below epsilon
epsilon = 0.01
generation_counter = 0
while True:
    generation_counter += 1
    
    #u_t: expected payoff per action at time t.
    # shape: 2x1
    u_t = np.matmul(reward[:,:,0],theta)
    
    #average expected payoff of the whole population
    # shape: 1x1
    u_t_star = np.matmul(np.transpose(theta),u_t)
    
    #theta_change: change in fraction of agents playing action a at time t
    # shape: 2x1
    # note: np.multiply is an element-wise multiplication
    # To check correctness: both elements should have the same absolute value
    theta_change = np.multiply(theta, u_t - u_t_star)
    
    #update population
    theta = theta + theta_change 
    #print "theta_change: " + str(np.transpose(theta_change))
    #check stopping condition
    if abs(theta_change[0]) < epsilon:
        break
    
print 'Final Player population: ' + str(np.transpose(theta))
print 'Total number of generations: ' + str(generation_counter)
