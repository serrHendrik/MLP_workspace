# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:11:01 2019

@author: Emile Breyne
         Andreas Stieglitz
         Hendrik Serruys

Replicator Dynamics

Implementation closely follows section 7.7 from Multiagent Systems (Shoham)

Note: the implementation is extended for non-symmetric games like the Matching Pennies game.
        This is achieved by maintaining two different kinds of players.

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Adding a step size slows down the redistribution of the population, and also prevents impossible populations (e.g. theta_p1 = [1.2 , -0.2])
step_size = 0.01

# Reward matrix: reward[action_player_0][action_player_1][reward_player_x]
#         Rock Paper Scissor
# Rock
# Paper
# Scissor
reward_matrix = np.array([[[0,0], [-1,1], [1,-1]],
                          [[1,-1], [0,0], [-1,1]],
                          [[-1,1], [1,-1], [0,0]]])

generations = 1000
learning_trajectories = 10

#Plot
fig = plt.figure()
ax = fig.add_subplot(111)
# Transform matrix
T = np.matrix([[1,-0.5],[-1,-0.5],[0,1]])
T[:,0] = T[:,0] / np.sqrt(1+1)
T[:,1] = T[:,1] / np.sqrt(0.5**2 + 0.5**2 + 1)


for _ in range(0,learning_trajectories):
    #theta: Action a -> proportion of players playing action a at time t
    #theta_p0 / theta_p1: theta for population of player0 / player1
    #shape: 2x1
    #initialize fractions randomly:
    
    theta_p0 = np.matrix([[random.random()], [random.random()], [random.random()]])
    theta_p0 = theta_p0 / np.sum(theta_p0)
    
    theta_p1 = theta_p0
    
    #print 'Initial theta_p0: ' + str(np.transpose(theta_p0))
    #print 'Initial theta_p1: ' + str(np.transpose(theta_p1))
    
    #plotting
    #The nth row represents the population density at generation n
    x = np.zeros((generations,3))
    
    for i in range(0,generations):
        mean_payoff_p0 = np.matmul(reward_matrix[:,:,0],theta_p1)
        mean_payoff_p1 = np.transpose(np.matmul(np.transpose(theta_p0),reward_matrix[:,:,1]))
        
        diff_theta_p0 = np.multiply(theta_p0, mean_payoff_p0 - np.matmul(np.transpose(theta_p0),mean_payoff_p0) )
        diff_theta_p1 = np.multiply(theta_p1, mean_payoff_p1 - np.matmul(np.transpose(theta_p1),mean_payoff_p1) )
        
        theta_p0 += diff_theta_p0 * step_size
        theta_p1 += diff_theta_p1 * step_size
    
        x[i,:] = theta_p0.ravel()

    #end for
    
    x_ = np.matmul(x,T)
    plt.plot(x_[:,0],x_[:,1],linestyle="solid")


#ax.set_xticks([])
#ax.set_yticks([])
ax.set_aspect('equal')
#plt.grid()
#plt.show()


# Phase plot

grid_dimension = 10
grid_dim = 10.2
plot_arrow_scaling = 1/10.0
offset = np.transpose(np.array([[0,1,0]]))

for i in range(0,grid_dimension + 1):
    for j in range(1,grid_dimension + 1):
        ii = i + 0.1
        jj = j + 0.1
        theta_p0 = offset + ii / grid_dim * np.sqrt(2) * T[:,0] + jj / grid_dim * np.sqrt(3)/2.0 * T[:,1]
        if (theta_p0[0,0] > 0 and theta_p0[1,0] > 0):
            mean_payoff_p0 = np.matmul(reward_matrix[:,:,0],theta_p0)
            diff_theta_p0 = np.multiply(theta_p0, mean_payoff_p0 - np.matmul(np.transpose(theta_p0),mean_payoff_p0) )
            theta_p0 += diff_theta_p0 * step_size
    
            #add arrow to the plot
            diff_ = np.matmul(np.transpose(diff_theta_p0),T)
            theta_ = np.matmul(np.transpose(theta_p0),T)
    
            plt.arrow(theta_[0,0],theta_[0,1],diff_[0,0]*plot_arrow_scaling,diff_[0,1]*plot_arrow_scaling,head_width=0.02, head_length=0.03)


#plt.grid()
plt.show()
