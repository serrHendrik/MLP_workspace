# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:31:16 2019


"""


import numpy as np
from agent_abstract import agent_abstract


"""
The agent_FR is a Free Riding agent. 

"""
class agent_FR(agent_abstract):
    
    def __init__(self, network, player, nb_rows, nb_cols, nb_players, state_size, action_size):
        print("\nPlayer " + str(player) + " is a FREE RIDER AGENT\n")
        super().__init__(network, player, nb_rows, nb_cols, nb_players, state_size, action_size)
    
    
    def calc_subjective_reward(self, rewards):  
        return rewards[self.player - 1]
        
    
    """
    FR Agents are pretty selfish..
    """
    def q_values_to_action(self, q_values):
        return np.argmax(q_values)