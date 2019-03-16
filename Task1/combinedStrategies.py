# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:21:54 2019

@author: emile
"""
import random
import numpy as np


###
# Q-learning
##
def Qselect_action(player,k):
    action = Pi[player]

    prob_action = k**Q[player][action] / (k**Q[player][0] + k**Q[player][1])
    
    if random.random() > prob_action:
        # explore a random action (uniform chance)
        action = int(round(random.random()))
        counter_rand_actions[player] += 1
        
        
    
    visits[player][action] += 1
    return action

# Update Q and V values
    #input: player x action x reward
def Qupdate(player,action,reward):
    
    # determine alpha
    #alpha = 1 / visits[player][action]
    alpha = 0.1
    
    # Update Q, V and Pi
    Q[player][action] = (1 - alpha)*Q[player][action] + alpha*(reward + gamma*V[player])
    if Q[player][0] >= Q[player][1]:
        V[player] = Q[player][0]
        Pi[player] = 0
    else:
        V[player] = Q[player][1]
        Pi[player] = 1


###PLA
#
#

def PLAselect_action(player):
    if random.random() <= p[player][0]:
        return 0
    else:
        return 1

def PLAupdate(player,action,reward):
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
n = 1000
#games: number of games played
games = 10000

#statistics PLA
final_prob_counter_PLA = np.array([[0.0,0.0],[0.0,0.0]])
###
#statistics Q-learning
final_policy_counter_Q = np.array([[0,0],[0,0]])

for episode in range(0,games):
    #PLA setup
    
    # p (probability matrix): Player x Action -> probability 
    init_p1 = random.random()
    init_p2 = random.random()
    p = np.array([[init_p1,1-init_p1] , [init_p2,1-init_p2]])
    #print 'Initial Probs: ' + str(p)


    #Q-learning setup
       #Q-learning
    # Initialisation
        # gamma: discount factor which determines the importance of future rewards
        # 0 =< gamma < 1
        # For gamma = 0, the agent is "myopic" (short sighted) and only considers the current reward.
    gamma = 0
        # inverse of learning rate: 
        # alpha(player,action) = 1 / visits(player,action)
        # visits(p,a) is incremented every time player p uses action a
    visits = [[0,0],[0,0]] 
        # Q-function: Player x Action -> Q-value
        # Note: No states needed as there is only 1 state.
        # Note: "High initial values, also known as "optimistic initial conditions", can encourage exploration: no matter what action is selected, the update rule will cause it to have lower values than the other alternative, thus increasing their choice probability." - source: https://en.wikipedia.org/wiki/Q-learning
    q_init = 100.0
    Q = [[q_init,q_init], [q_init,q_init]]
        # Strategy / Policy function: Player -> Action
        # Note: Action codes: (C)olaborate = 0, (D)efect = 1
    Pi = [int(round(random.random())),int(round(random.random()))]
    #print "Initial Policy: " + str(Pi)
        # V-function: Player -> V-value
        # Note: V(p) = max_a Q(p,a)
    V = [max(Q[0]),max(Q[1])]
    
    # Select an action for a player
    # Note: Player codes: player_0 = 0, player_1 = 1
    # Note: prob_action is the probability of actually picking the action.
    #       prob_action increases towards 1 with increasing k
    #       (formula derived from ML & Inductive Inference (H. Blockeel), p. 265)
    counter_rand_actions = [0,0];

    #play n rounds
    for episode in range(0,n):
        # Q-learning
        # k is an exploration variable used in Qselect_action
        k = episode / 1000.0 + 1.0      
        
        #player 0 : uses Q-learning Algorithm
        a_p0 = Qselect_action(0,k)
        #player 1 : uses PLA
        a_p1 = PLAselect_action(1)
        

        
        if game_code == 0:
            # Prisoner's Dilemma
            r = reward_pd[a_p0][a_p1]
            r_p1 = reward_pd[a_p0][a_p1][1] - reward_pd[a_p0][abs(a_p1 - 1)][1]
        else:
            # Matching Pennies
            r = reward_mp[a_p0][a_p1]        
            r_p1 = reward_mp[a_p0][a_p1][1] - reward_mp[a_p0][abs(a_p1 - 1)][1]  
        
        Qupdate(0,a_p0,r[0])
        Qupdate(1,a_p1,r[1])
        
        PLAupdate(1,a_p1,r_p1)
        
    #registrate final probabilities
    final_prob_counter_PLA += p
    
    #registrate final policy
    final_policy_counter_Q[Pi[0]][Pi[1]] += 1
    
    # =============================================================================
    # #calculate results
    # print '*** RESULTS ***'
    # print 'Probabilities: ' + str(p)
    # =============================================================================
    
print(" Games played: " + str(games))
print('*** player 0 (Q-learning) Results ***')
print( " final_policy_counter \n" + str(final_policy_counter_Q))
print( " In percentage: \n" + str(100 * final_policy_counter_Q / (games * 1.0)))


print('*** player 1 (PLA) Results ***')
print(" Average probabilities for player 1: \n" + str(final_prob_counter_PLA[1] / (games * 1.0)))


