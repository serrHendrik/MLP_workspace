# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:46:33 2019

@author:
    
Combined Q-learning with opponent modelling
    
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from fictitiousPlayerRPS import fictitiousPlayerRPS
from qomPlayerRPS import qomPlayerRPS

        

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
    
    def __init__(self, episodes = 10000, player1 = qomPlayerRPS(), player2 = qomPlayerRPS()):
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
            T = 10.0 / float(np.min([ep+1,1000]))
            
            if (isinstance(self.p1,qomPlayerRPS)):
                play_p1 = self.p1.play(T)
            elif (isinstance(self.p1,fictitiousPlayerRPS)):
                play_p1 = self.p1.play()
            else:
                print("ERROR: Player 1 is an instance of an unknown type.")
                return -1
            
            if (isinstance(self.p2,qomPlayerRPS)):
                play_p2 = self.p2.play(T)
            elif (isinstance(self.p2,fictitiousPlayerRPS)):
                play_p2 = self.p2.play()
            else:
                print("ERROR: Player 2 is an instance of an unknown type.")
                return -1
            
            #print("Player moves: " + str(play_p1) + " and " + str(play_p2))
            payoff_p1, payoff_p2 = self.reward_bimatrix[play_p1,play_p2]
            self.p1.update(play_p1,play_p2,payoff_p1)
            self.p2.update(play_p2,play_p1,payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results

alpha = 0.1
max_init = 10
p1 = qomPlayerRPS(alpha = 0.1, initialBeliefs = [random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
p2 = qomPlayerRPS(alpha = 0.1, initialBeliefs = [random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
p3 = fictitiousPlayerRPS([random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])


game = RPSgame(player1 = p1, player2 = p3) 
results = game.play()
print(results)
print("Row player: " + str(results.sum(axis=1) / sum(sum(results))))
print("Column player: " + str(results.sum(axis=0) / sum(sum(results))))

if (isinstance(game.p1,qomPlayerRPS)):
    print("Random number of actions for player p1: " + str(game.p1.counter) + " out of a total of " + str(game.episodes))
if (isinstance(game.p2,qomPlayerRPS)):
    print("Random number of actions for player p2: " + str(game.p2.counter) + " out of a total of " + str(game.episodes))

print("Total Payoff for p1: " + str(game.p1.total_reward) + " and for p2: " + str(game.p2.total_reward))




#Visualisation
# 3D to 2D Transform matrix
T = np.matrix([[1,-0.5],[-1,-0.5],[0,1]])
# Normalize vectors
T[:,0] = T[:,0] / np.sqrt(1+1)
T[:,1] = T[:,1] / np.sqrt(0.5**2 + 0.5**2 + 1)

#Plotted Triagle is
#   Scissors
# Paper     Rock
fig = plt.figure(figsize=(10,5))

# Plot beliefs of player 2
ax1 = fig.add_subplot(121)
ax1.grid()
ax1.axis('equal')
ax1.set_title("Player2's beliefs about Player1")
#plot contour of triangle
corners = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,0]])
corners_tf = np.matmul(corners,T)
ax1.plot(corners_tf[:,0],corners_tf[:,1],linestyle='solid',color="black")

# Plot history of probability of opponent's mixed strategy
timeline = np.matmul(game.p2.beliefs_timeline,T)
edge = int(timeline.shape[0]/3)
timeline1=  timeline[0:edge,:]
timeline2=  timeline[edge:2*edge,:]
timeline3=  timeline[2*edge:3*edge,:]
ax1.plot(timeline1[:,0],timeline1[:,1],'x-',color="tomato")
ax1.plot(timeline2[:,0],timeline2[:,1],'x-',color="red")
ax1.plot(timeline3[:,0],timeline3[:,1],'x-',color="darkred")

# Plot beliefs of player 1
ax2 = fig.add_subplot(122)
ax2.grid()
ax2.axis('equal')
ax2.set_title("Player1's beliefs about Player2")
#plot contour of triangle
ax2.plot(corners_tf[:,0],corners_tf[:,1],linestyle='solid',color="black")

# Plot history of probability of opponent's mixed strategy
timeline = np.matmul(game.p1.beliefs_timeline,T)
edge = int(timeline.shape[0]/3)
timeline1=  timeline[0:edge,:]
timeline2=  timeline[edge:2*edge,:]
timeline3=  timeline[2*edge:3*edge,:]
ax2.plot(timeline1[:,0],timeline1[:,1],'x-',color="tomato")
ax2.plot(timeline2[:,0],timeline2[:,1],'x-',color="red")
ax2.plot(timeline3[:,0],timeline3[:,1],'x-',color="darkred")

ax1.axis('off')
ax1.grid(b=None)
ax2.axis('off')
ax2.grid(b=None)
plt.show()
