# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:47:46 2019


"""


import random
import numpy as np
import math

# Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
#         Rock Paper Scissor
# Rock
# Paper
# Scissor
reward_matrix = [[[0,0], [-1,1], [1,-1]],
                 [[1,-1], [0,0], [-1,1]],
                 [[-1,1], [1,-1], [0,0]]]
#n = number of episodes played in a game
n = 10000
#games: number of games played
games = 1000

#statistics
final_policy_counter = np.array([[0,0,0],[0,0,0],[0,0,0]])
###


# Select an action for a player
# Note: Player codes: player_0 = 0, player_1 = 1
# Note: prob_action is the probability of actually picking the action.
#       prob_action increases towards 1 with increasing k
#       (formula derived from ML & Inductive Inference (H. Blockeel), p. 265)
def select_action(player,k):
    action = Pi[player]
    prob_action = k**Q[player][action] / (k**Q[player][0] + k**Q[player][1] + k**Q[player][2])
    
    if random.random() > prob_action:
        # explore a random action (action uniformly chosen)
        action = math.trunc((3*random.random()))
        counter_rand_actions[player] += 1
        
    visits[player][action] += 1
    return action

# Update Q and V values
    #input: player x action x reward
def update(player,action,reward):
    
    # determine alpha
    #alpha = 1 / visits[player][action]
    alpha = 0.1
    
    # Update Q, V and Pi
    Q[player][action] = (1 - alpha)*Q[player][action] + alpha*(reward + gamma*V[player])
    if Q[player][action] >= V[player]:
        V[player] = Q[player][action]
        Pi[player] = action

for _ in range(0,games):
    #Q-learning
    # Initialisation
        # gamma: discount factor which determines the importance of future rewards
        # 0 =< gamma < 1
        # For gamma = 0, the agent is "myopic" (short sighted) and only considers the current reward.
    gamma = 0.0
        # inverse of learning rate: 
        # alpha(player,action) = 1 / visits(player,action)
        # visits(p,a) is incremented every time player p uses action a
    visits = [[0,0,0],[0,0,0]] 
        # Q-function: Player x Action -> Q-value
        # Note: No states needed as there is only 1 state.
        # Note: "High initial values, also known as "optimistic initial conditions", 
        # can encourage exploration: no matter what action is selected, 
        # the update rule will cause it to have lower values than the other alternative, 
        # thus increasing their choice probability." 
        # - source: https://en.wikipedia.org/wiki/Q-learning
    q_init = 100.0
    Q = [[q_init,q_init, q_init], [q_init,q_init,q_init]]
        # Strategy / Policy function: Player -> Action
        # Note: Action codes: Rock = 0, Paper = 1, Scissors = 2
    Pi = [int(math.trunc(3*random.random())),int(math.trunc(3*random.random()))]
    #print "Initial Policy: " + str(Pi)
        # V-function: Player -> V-value
        # Note: V(p) = max_a Q(p,a)
    V = [max(Q[0]),max(Q[1])]
    
        # Keep a counter to check upon the amount of random actions per player (see select_action())
    counter_rand_actions = [0,0];
    
    #play n rounds
    for episode in range(0,n):
        
        # k is an exploration variable used in select_action.
        # based on counter_rand_actions, tune this value to a desired degree of exploration
        k = episode / 0.1 + 1.0
        
        a_p0 = select_action(0,k)
        a_p1 = select_action(1,k)
        
        r = reward_matrix[a_p0][a_p1]        
        
        update(0,a_p0,r[0])
        update(1,a_p1,r[1])
    #end for
    
    #registrate final policy
    final_policy_counter[Pi[0]][Pi[1]] += 1
    
    
    #calculate results
# =============================================================================
#     print('*** RESULTS ***')
#     print("visits: " + str(visits))
#     print("Q: " + str(Q))
#     print("Final Policy: " + str(Pi))
#     print('')
#     print('*** Extra ***')
#     print("Random actions per player: " + str(counter_rand_actions))
# =============================================================================

print(" *** Results ***")
print( " Games played: " + str(games))
print( " final_policy_counter \n" + str(final_policy_counter))
print( " In percentage: \n" + str(100.0 * final_policy_counter / (games * 1.0)))

# Conclusion: 
# Similar to Pennies Game.
# All strategies are Pareto optimal (zero-sum game)
# Nash Equilibrium is (0.33,0.33,0.33).


     