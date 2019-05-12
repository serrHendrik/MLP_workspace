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
from FPPlayer import FPPlayer

        
class FPgame:
    # Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
    #         Rock Paper Scissor
    # Rock
    # Paper
    # Scissor
    reward_bimatrix = np.array([[[0,0], [-1,1], [1,-1]],
                                [[1,-1], [0,0], [-1,1]],
                                [[-1,1], [1,-1], [0,0]]])
    reward_matrix = reward_bimatrix[:,:,0]
    
    def __init__(self, player1 = FPPlayer(), player2 = FPPlayer()):
        self.p1 = player1
        self.p2 = player2
        self.results = np.zeros([3,3])
        
    def play(self, episodes = 20000):
        for ep in range(episodes):
            play_p1 = self.p1.play()
            play_p2 = self.p2.play()
            payoff_p1, payoff_p2 = self.reward_bimatrix[play_p1,play_p2]
            self.p1.update(play_p1,play_p2, payoff_p1)
            self.p2.update(play_p2,play_p1, payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results
   
    
# PLAY
        
max_init = 10
p1 = FPPlayer([random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
p2 = FPPlayer([random.randint(1,max_init),random.randint(1,max_init),random.randint(1,max_init)])
game = FPgame(p1,p2) 
results = game.play()

print(results)
print(str(results / float(sum(sum(results)))))
print("Moves Row player: " + str(results.sum(axis=1) / float(sum(sum(results)))))
print("Moves Column player: " + str(results.sum(axis=0) / float(sum(sum(results)))))
print("Score Row player: " + str(game.p1.total_reward))
print("Score Column player: " + str(game.p2.total_reward))



# VISUALIZE

plt.figure(figsize=(10,5))
plt.plot(game.p1.total_reward_timeline,color="blue")
plt.plot(game.p2.total_reward_timeline,color="orange")
plt.ylabel("Total accumulated reward")
plt.xlabel("Time [episodes]")
plt.legend(["Player 1", "Player 2"])
plt.grid()
plt.rcParams.update({'font.size': 16})
plt.savefig("Images/FP_total_reward_temp.png")
plt.show()

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
timeline = np.matmul(game.p1.beliefs_timeline,T)
edge = int(timeline.shape[0]/3)
timeline1=  timeline[0:edge,:]
timeline2=  timeline[edge:2*edge,:]
timeline3=  timeline[2*edge:3*edge,:]
ax.plot(timeline1[:,0],timeline1[:,1],'x-',color="tomato")
ax.plot(timeline2[:,0],timeline2[:,1],'x-',color="red")
ax.plot(timeline3[:,0],timeline3[:,1],'x-',color="darkred")
plt.axis('off')
plt.grid(b=None)
plt.savefig("Images/FP_belief_temp.png")
plt.show()



