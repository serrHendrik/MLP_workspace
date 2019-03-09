# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:48:47 2019

@author: Hendrik Serruys
"""

"""
Change the game_code (CTRL+F) to change the game:
The Prisoner's Dilemma Game (0)
The Matching Pennies Game (1)


PLA: Probabilistic Learning Automata

Implementation based on PROBABILISTIC LEARNING AUTOMATA (Kehagias), starting from section V

"""
import random


# p (probability matrix): Player x Action -> probability 
init_p1 = random.random()
init_p2 = random.random()
p = [[init_p1,1-init_p1] , [init_p2,1-init_p2]]
print 'Initial Probs: ' + str(p)

def select_action(player):
    if random.random() <= p[player][0]:
        return 0
    else:
        return 1

def update(player,action,reward):
    if reward < 0:
        #punishment
        learning_rate = abs(reward) / 1000.0
        p[player][action] *= (1-learning_rate)
    else:
        #reward
        learning_rate = reward / 1000.0
        p[player][action] += (1-p[player][action])*learning_rate
        
    #update alternative action probability
    alternative = abs(action - 1)
    p[player][alternative] = 1 - p[player][action]
    

###
# game_code: Prisoner's Dilemma (0)
#            Matching Pennies (1)
game_code = 0
# Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
T = 5
R = 3
S = 0
P = 1
reward_pd = [[[R,R],[S,T]] , [[T,S],[P,P]]]
reward_mp = [[[1,-1],[-1,1]] , [[-1,1],[1,-1]]]
#n = number of episodes played
n = 50000
###


#play n rounds
for episode in range(0,n):
    
    a_p0 = select_action(0)
    a_p1 = select_action(1)
    
    if game_code == 0:
        # Prisoner's Dilemma
        r_p0 = reward_pd[a_p0][a_p1][0] - reward_pd[abs(a_p0 - 1)][a_p1][0]
        r_p1 = reward_pd[a_p0][a_p1][1] - reward_pd[a_p0][abs(a_p1 - 1)][1]
    else:
        # Matching Pennies
        r_p0 = reward_mp[a_p0][a_p1][0] - reward_mp[abs(a_p0 - 1)][a_p1][0]
        r_p1 = reward_mp[a_p0][a_p1][1] - reward_mp[a_p0][abs(a_p1 - 1)][1]  
    
    update(0,a_p0,r_p0)
    update(1,a_p1,r_p1)
    

#calculate results
print '*** RESULTS ***'
print 'Probabilities: ' + str(p)



