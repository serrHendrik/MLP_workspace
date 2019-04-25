# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:49:53 2019

@authors: 
    Emile Breyne
    Andreas Stieglitz
    Hendrik Serruys
"""

"""
Replicator Dynamics

Implementation closely follows section 7.7 from Multiagent Systems (Shoham)

Note: the implementation is extended for non-symmetric games like the Matching Pennies game.
        This is achieved by maintaining two different kinds of players.

"""
import random
import numpy as np
import matplotlib.pyplot as plt


# Adding a step size slows down the redistribution of the population, and also prevents impossible populations (e.g. theta_p1 = [1.2 , -0.2])
step_size = 0.007

###
# game_code: Prisoner's Dilemma (0)
#            Matching Pennies (1)
game_code = 1
learning_trajectories = 10
# Reward / Utility matrix: reward[action_player_0][action_player_1][reward_player_x]
T = 5
R = 3
S = 0
P = 1
reward_pd = np.array([[[R,R],[S,T]] , [[T,S],[P,P]]])
reward_mp = np.array([[[1.0,-1.0],[-1.0,1.0]] , [[-1.0,1.0],[1.0,-1.0]]])
reward = reward_pd if game_code == 0 else reward_mp

generations = 1000


for lt in range(0,learning_trajectories):
    #theta: Action a -> proportion of players playing action a at time t
    #theta_p0 / theta_p1: theta for population of player0 / player1
    #shape: 2x1

    #using handpicked initial fractions allows for equidistant dynamics to be drawn
    if game_code == 0:
        init_fraction0 = 0.95+float(lt)/float(20*learning_trajectories)
        init_fraction1 = 1.95- init_fraction0
    else:
        init_fraction0 = float(lt)/float(learning_trajectories)
        init_fraction1 = float(lt)/float(learning_trajectories)
    theta_p0 = np.matrix([[init_fraction0] , [1 - init_fraction0]])
    theta_p1 = np.matrix([[init_fraction1] , [1 - init_fraction1]])
    #print 'Initial theta_p0: ' + str(np.transpose(theta_p0))
    #print 'Initial theta_p1: ' + str(np.transpose(theta_p1))
    
    #plotting
    x = np.zeros(generations)
    y = np.zeros(generations)
    
    for i in range(0,generations):
        
        mean_payoff_p0 = np.matmul(reward[:,:,0],theta_p1)
        mean_payoff_p1 = np.transpose(np.matmul(np.transpose(theta_p0),reward[:,:,1]))
        
        diff_theta_p0 = np.multiply(theta_p0, mean_payoff_p0 - np.matmul(np.transpose(theta_p0),mean_payoff_p0) )
        diff_theta_p1 = np.multiply(theta_p1, mean_payoff_p1 - np.matmul(np.transpose(theta_p1),mean_payoff_p1) )
        
        theta_p0 += diff_theta_p0 * step_size
        theta_p1 += diff_theta_p1 * step_size
    
        x[i] = theta_p0[0]
        y[i] = theta_p1[0]
        
    #print 'Final Player0 population: ' + str(np.transpose(theta_p0))
    #print 'Final Player1 population: ' + str(np.transpose(theta_p1))
    
    plt.plot(x,y,linestyle="solid",color="darkred")

#plt.axis([0, 1, 0, 1]);
#plt.grid()
#plt.show()

# Phase plot
grid_dimension = 11
plot_arrow_scaling = 1/10.0


theta0 = [0,0]
theta1 = [0,0]
for t0 in range(0,grid_dimension):
    for t1 in range(0,grid_dimension):
        theta0[0] = t0/float(grid_dimension-1)
        theta0[1] = 1-t0/float(grid_dimension-1)
        theta1[0] = t1/float(grid_dimension-1)
        theta1[1] = 1-t1/float(grid_dimension-1)
        #calculate changes
        Ay=((np.matmul(reward[:,:,0],theta1)))
        theta0_change = theta0[0]*(Ay[0]-np.matmul(np.transpose(theta0),Ay))
        Bx=(np.transpose((np.matmul(np.transpose(theta0),reward[:,:,1]))))
        theta1_change = theta1[0]*(Bx[0]-np.matmul(np.transpose(theta1),Bx))

        #add arrow to the plot
        plt.arrow(theta0[0],theta1[0],theta0_change*plot_arrow_scaling,theta1_change*plot_arrow_scaling,head_width=0.02, head_length=0.03)




plt.axis([0, 1, 0, 1]);
move = "Collaborate" if game_code == 0 else "Heads"
plt.xlabel("Player 1: Odds of playing "+move)
plt.ylabel("Player 2: Odds of playing "+move)
plt.grid()
plt.show()
