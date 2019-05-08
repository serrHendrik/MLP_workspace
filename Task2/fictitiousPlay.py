# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:42:03 2019

@author: emile

Fictitious play
Based on book Multiagent Systems: Algorithmic, Game-Theoretic, and logical functions (Shoham)
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from fictitiousPlayerRPS import fictitiousPlayerRPS

        
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
    
    def __init__(self, player1 = fictitiousPlayerRPS(), player2 = fictitiousPlayerRPS()):
        self.p1 = player1
        self.p2 = player2
        self.results = np.zeros([3,3])
        
    def play(self, episodes = 1000):
        for ep in range(episodes):
            play_p1 = self.p1.play()
            play_p2 = self.p2.play()
            payoff_p1, payoff_p2 = self.reward_bimatrix[play_p1,play_p2]
            self.p1.update(play_p1,play_p2, payoff_p1)
            self.p2.update(play_p2,play_p1, payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results
    
max_init = 10
p1 = fictitiousPlayerRPS([random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
p2 = fictitiousPlayerRPS([random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
game = RPSgame(p1,p2) 
#game = RPSgame()
results = game.play()
print(results)
print("Row player: " + str(results.sum(axis=1) / sum(sum(results))))
print("Column player: " + str(results.sum(axis=0) / sum(sum(results))))
#Findings: fictitious play very influencable by initial beliefs
#What when beliefs of 2 actions are equal? (now just maintain same strategy)       



#Visualising
fig = plt.figure(figsize=(10,5))
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
timeline = np.matmul(game.p1.beliefs_timeline,T)
plt.plot(timeline[:,0],timeline[:,1],'x-',color="darkred")
plt.axis('off')
plt.grid(b=None)
plt.show()





