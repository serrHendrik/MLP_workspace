# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:47:57 2019


"""

import random
import numpy as np
from agent_abstract import agent_abstract

"""
The Agent_IA is Inequity Averse (IA), both advantageous (AIA) and disadvantageous (DIA)
"""
class agent_IA(agent_abstract):
    
    def __init__(self, network, player, nb_rows, nb_cols, nb_players, state_size, action_size):
        print("\nPlayer " + str(player) + " is an INEQUITY ADVERSE AGENT\n")
        super().__init__(network, player, nb_rows, nb_cols, nb_players, state_size, action_size)
        #e values and subjective rewards as defined in (Hughes, Leibo, Tuyls et al. 2018)
        self.g = 0.9    #gamma used to calculate e-value
        self.l = 1      #lamda used to calculate e-value
        self.e_values = np.zeros(nb_players)
        self.a = 4.0    #alpha used to calculate subjective rewards
        self.b = 1.0   #beta used to calculate subjective rewards
    
    
    
    def observe(self, rewards, players, apples, done):
        self.update_e_values(rewards)
        super().observe(rewards,players,apples,done)
        
        
    def calc_subjective_reward(self, rewards):   
        # Update e values
        self.update_e_values(rewards)
        
        # Calculdate subjective reward
        player_index = self.player - 1
        envy = 0.0
        guilt = 0.0
        for j in range(0,self.nb_players):
            if self.e_values[j] > self.e_values[player_index]:
                envy += self.e_values[j] - self.e_values[player_index]
            else:
                guilt += self.e_values[player_index] - self.e_values[j]
        
        envy = envy * self.a / float(self.nb_players - 1)
        guilt = guilt * self.b / float(self.nb_players - 1)
        
        sr = rewards[player_index] - envy - guilt
        return sr
    
    
    def update_e_values(self,rewards):
        self.e_values = self.g * self.l * self.e_values + rewards
        
    
    """
    
    The Q-values only provide a guidance towards apples (increasing the agent's short term reward).
    In other words, our Q-values are designed to reflect the agent's impulses / emotions towards short term gratification.
    In this method however, the agent will reflect on its impulses. As Marcus Aurelius would put it:
        “Waste no more time arguing about what a good man should be. Be one.”
    Using envy and guilt, the agent will choose its course of action.
    """
    def q_values_to_action(self, q_values):
        player_index = self.player - 1
        envy = 1.0
        guilt = 1.0
        for j in range(0,self.nb_players):
            if self.e_values[j] > self.e_values[player_index]:
                envy += self.e_values[j] - self.e_values[player_index]
            else:
                guilt += self.e_values[player_index] - self.e_values[j]
        
        envy = envy * self.a / float(self.nb_players - 1)
        guilt = guilt * self.b / float(self.nb_players - 1)
        
        #Probability of acting envious
        prob_envy = envy / (envy + guilt)
        
        #With prob_envy, choose action with highest Q-value
        #When guilt overwhelms, abstain from eating apples
        if random.rand() <= prob_envy:
            action_index = np.argmax(q_values)
        else:
            action_index = np.argmin(q_values)
        
        return action_index
    
    