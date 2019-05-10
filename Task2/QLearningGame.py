# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:03:26 2019

@author: serru
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from QLearningPlayer import QLearningPlayer

        
class QLearningGame:
    # Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
    #         Rock Paper Scissor
    # Rock
    # Paper
    # Scissor
    reward_bimatrix = np.array([[[0,0], [-1,1], [1,-1]],
                                [[1,-1], [0,0], [-1,1]],
                                [[-1,1], [1,-1], [0,0]]])
    reward_matrix = reward_bimatrix[:,:,0]
    
    def __init__(self, episodes = 20000, player1 = QLearningPlayer(), player2 = QLearningPlayer()):
        self.episodes = episodes
        self.p1 = player1
        self.p2 = player2
        self.results = np.zeros([3,3])
        
    def play(self):
        for ep in range(self.episodes):
            #T: Temperature in Boltzmann Exploration
            # T > 0
            # For T -> +inf, an agent will choose an action at random (exploration)
            # For T -> 0, an agent will choose the action with largest EV (exploitation)
            T = 10.0 / float(np.min([ep+1,120]))
            play_p1 = self.p1.play(T)
            play_p2 = self.p2.play(T)
            payoff_p1, payoff_p2 = self.reward_bimatrix[play_p1,play_p2]
            self.p1.update(play_p1,play_p2, payoff_p1)
            self.p2.update(play_p2,play_p1, payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results
    
    
    
# PLAY

results = np.zeros([3,3])
nb_games = 1
for _ in range(nb_games):
    game = QLearningGame() 
    results += game.play()

results /= float(nb_games)

print("Average results over " + str(nb_games) + " of games:")
print(results)
print("Moves Row player: " + str(results.sum(axis=1) / sum(sum(results))))
print("Moves Column player: " + str(results.sum(axis=0) / sum(sum(results))))
print("About the last game:")
print("Score Row player: " + str(game.p1.total_reward))
print("Score Column player: " + str(game.p2.total_reward))
print("Random number of actions for player p1: " + str(game.p1.counter) + " out of a total of " + str(game.episodes))
print("Player p1 final Q-values:\n" + str(game.p1.Q))
print("Player p1 final probabilities:\n" + str(game.p1.probs))


#VISUALIZE

plt.figure(figsize=(10,5))
plt.plot(game.p1.total_reward_timeline,color="blue")
plt.plot(game.p2.total_reward_timeline,color="orange")
plt.ylabel("Total accumulated reward")
plt.xlabel("Time [episodes]")
plt.grid()
plt.legend(["Player 1", "Player 2"])
plt.rcParams.update({'font.size': 16})
plt.savefig("Images/QLearning_total_reward.png")
plt.show()

"""
#Visualising
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
# Transform matrix
T = np.matrix([[1,-0.5],[-1,-0.5],[0,1]])
T[:,0] = T[:,0] / np.sqrt(1+1)
T[:,1] = T[:,1] / np.sqrt(0.5**2 + 0.5**2 + 1)

#plot contour of triangle
corners = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,0]])
corners_tf = np.matmul(corners,T);
plt.plot(corners_tf[:,0],corners_tf[:,1],linestyle='solid',color="black")

# Plot history of probability of opponent's mixed strategy
timeline = np.matmul(game.p1.probs_timeline,T)
plt.plot(timeline[1000:,0],timeline[1000:,1],'x-',color="darkred")
plt.axis('off')
plt.grid(b=None)
plt.show()
"""