
# -*- coding: utf-8 -*-
"""
@author: Andreas Stieglitz
"""

"""
Replicator Dynamics
Implementation closely follows section 3.2.2 from ANALYZING REINFORCEMENT LEARNING ALGORITHMS USING EVOLUTIONARY GAME THEORY (Daan Bloembergen)

"""
import random
import numpy as np
import matplotlib.pyplot as plot



###
# game_code: Prisoner's Dilemma (0)
#            Matching Pennies (1)
game_code = 1
grid_dimension = 11
plot_arrow_scaling = 1/10.0

# Reward / Utility matrix: reward[action_player_0][action_player_1][reward_player_x]
T = 5
R = 3
S = 0
P = 1
reward_pd = np.array([[[R,R],[S,T]] , [[T,S],[P,P]]])
reward_mp = np.array([[[1,-1],[-1,1]] , [[-1,1],[1,-1]]])
reward = reward_pd if game_code == 0 else reward_mp

theta0 = [0,0]
theta1 = [0,0]
for t0 in range(0,grid_dimension):
    for t1 in range(0,grid_dimension):
        theta0[0] = t0/float(grid_dimension-1)
        theta0[1] = 1-t0/float(grid_dimension-1)
        theta1[0] = t1/float(grid_dimension-1)
        theta1[1] = 1-t1/float(grid_dimension-1)
        #calculate changes
        Ay=((np.matmul(reward[0,:,:],theta1)))
        theta0_change = theta0[0]*(Ay[0]-np.matmul(np.transpose(theta0),Ay))
        Bx=((np.matmul(reward[1,:,:],theta0)))
        theta1_change = theta1[0]*(Bx[0]-np.matmul(np.transpose(theta1),Bx))

        #add arrow to the plot
        plot.arrow(theta0[0],theta1[0],theta0_change*plot_arrow_scaling,theta1_change*plot_arrow_scaling,head_width=0.02, head_length=0.03)

plot.grid()
plot.show()


    



