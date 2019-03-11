# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:49:53 2019

@author: Hendrik Serruys
"""

"""
Replicator Dynamics

Implementation closely follows section 7.7 from Multiagent Systems (Shoham)

Note: the implementation is extended for non-symmetric games like the Matching Pennies game.
        This is achieved by maintaining two different kinds of players.

"""
import random
import numpy as np

#theta: Action -> proportion of players playing action a at time t
#theta0 / theta1: theta for population of player0 / player1
#shape: 2x1
init_fraction0 = random.random()
init_fraction1 = random.random()
theta0 = np.matrix([[init_fraction0] , [1 - init_fraction0]])
theta1 = np.matrix([[init_fraction1] , [1 - init_fraction1]])
print 'Initial theta0: ' + str(np.transpose(theta0))
print 'Initial theta1: ' + str(np.transpose(theta1))

###
# game_code: Prisoner's Dilemma (0)
#            Matching Pennies (1)
game_code = 1
# Reward / Utility matrix: reward[action_player_0][action_player_1][reward_player_x]
T = 5
R = 3
S = 0
P = 1
reward_pd = np.array([[[R,R],[S,T]] , [[T,S],[P,P]]])
reward_mp = np.array([[[1.0,-1.0],[-1.0,1.0]] , [[-1.0,1.0],[1.0,-1.0]]])
reward = reward_pd if game_code == 0 else reward_mp
#epsilon: We generate new generations until the rate of change in the fraction of agents playing action a at time t is below epsilon
epsilon = 0.001
generation_counter = 0
for _ in range(0,1000):
    
     mF = np.matmul(reward[:,:,0],theta0)
    
     theta0 = np.multiply(theta0, mF - np.matmul(np.transpose(theta0),mF) )
     #print "Theta: " + str(np.transpose(theta0))

     generation_counter += 1

     #ux_t: expected payoff per action at time t for player x, given the choice of action of player x.
     #       In other words, ux_t[0] is the payoff we would have received if the entire population of player x played action 0.
     # shape: 2x1
     # Note: np.matmul(reward[:,:,0],theta1) is the conditional expected payoff, conditioning over the action of player 0.
     u0_t = np.transpose(np.matmul(np.transpose(theta0),reward[:,:,0]))
     #u1_t = np.matmul(reward[:,:,1],theta1)
     u1_t = np.transpose(np.matmul(np.transpose(theta1),reward[:,:,1]))

     #average expected payoff of the whole population
     # shape: 1x1
     u0_t_star = np.matmul(np.transpose(theta1),u0_t)
     u1_t_star = np.matmul(np.transpose(theta0),u1_t)

     #theta_change: change in fraction of agents playing action a at time t
     # shape: 2x1
     # note: np.multiply is an element-wise multiplication
     # To check correctness: both elements should have the same absolute value


     theta0_change = np.multiply(theta0,u0_t - u0_t_star)
     theta1_change = np.multiply(theta1,u1_t - u1_t_star)

     #update population
     theta0 = theta0 + theta0_change
     theta1 = theta1 + theta1_change
     #Some problems occurred with probabilities not summing to 1 and negative probabilities
     #both are explicitly handled in the code below
     if theta0[0]<0: theta0[0] = 0
     if theta1[0]<0: theta1[0] = 0
     theta0[1]=1-theta0[0]
     theta1[1]=1-theta1[0]
     #print "theta_changes: " + str(np.transpose(theta0_change)) + str(np.transpose(theta1_change))
     
     #check stopping condition
     if abs(theta0_change[0]) < epsilon and abs(theta1_change[0]) < epsilon:
         break

    
print 'Final Player0 population: ' + str(np.transpose(theta0))
print 'Final Player1 population: ' + str(np.transpose(theta1))
print 'Total number of generations: ' + str(generation_counter)
