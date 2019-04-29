# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:47:57 2019


"""

import numpy as np
from custom_DRL_agents import new_DQNAgent

class agent_IA:
    
    def __init__(self, player, nb_rows, nb_cols, nb_players, actions):
        print("\nPlayer " + str(player) + " is an INEQUITY ADVERSE AGENT\n")
        self.player = player
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        self.actions = actions
        
        self.network = new_DQNAgent(nb_actions=len(actions))
        
        
        #e values and subjective rewards as defined in (Hughes, Leibo, Tuyls et al. 2018)
        self.g = 0.9    #gamma used to calculate e-value
        self.l = 1      #lamda used to calculate e-value
        self.e_values = np.zeros(nb_players)
        self.a = 5.0    #alpha used to calculate subjective rewards
        self.b = 0.05   #beta used to calculate subjective rewards
        self.subj_rewards = np.zeros(nb_players)
        
        
    
    
    def next_action(self, players, apples):
        state = np.zeros((15,15))
        player_x, player_y = players[self.player-1]["location"]
        print("Current Player: " + str(self.player))
        print(str(players[self.player-1]))
        #print("player_x and player_y: " + str(player_x) + " " + str(player_y))
        #fill network input with apples on second channel
        for apple_x, apple_y in apples:
            #print("apple_x and apple_y original: " + str(apple_x) + " " + str(apple_y))
            if apple_x < player_x - 7:
                apple_x += self.nb_cols
            elif apple_x > player_x + 7:
                apple_x -= self.nb_cols
            
            if apple_y < player_y - 7:
                apple_y += self.nb_rows
            elif apple_y > player_y + 7:
                apple_y -= self.nb_rows
            #print("apple_x and apple_y transformed: " + str(apple_x) + " " + str(apple_y))
            inp_x = apple_x - player_x + 7 - 1
            inp_y = apple_y - player_y + 7 - 1
            #print("inp_x and inp_y: " + str(inp_x) + " " + str(inp_y))
            state[inp_y,inp_x] = 1.0
        
        #fill network ipnut with players on third channel
        for p in players:
            p_x, p_y = p["location"]
            if (type(p_x) == int):
                #handle endless field
                #print("p_x and p_y original: " + str(p_x) + " " + str(p_y))
                if p_x < player_x - 7:
                    p_x += self.nb_cols
                elif p_x > player_x + 7:
                    p_x -= self.nb_cols
                
                if p_y < player_y - 7:
                    p_y += self.nb_rows
                elif p_y > player_y + 7:
                    p_y -= self.nb_rows
                #print("p_x and p_y transformed: " + str(p_x) + " " + str(p_y))
                inp_x = p_x - player_x + 7 - 1
                inp_y = p_y - player_y + 7 - 1
                state[inp_y,inp_x] = 0.5
        
        # Exploit symmetry to help network train better
        # Rotate state so that player orientation is up
        orientation = players[self.player-1]["orientation"]
        
        if orientation == "right":
            state = np.rot90(state,k=1)
        elif orientation == "down":
            state = np.rot90(state,k=2)
        elif orientation == "left":
            state = np.rot90(state,k=3)
        
        #Feed to network
        action_index = self.network.act(states=state)
        return self.actions[action_index]
    
    def observe(self, rewards, terminal):
        subj_reward = self.calc_subj_reward(rewards)
        self.network.observe(reward=subj_reward, terminal=terminal)
    
    def calc_subj_reward(self, rewards):   

        # Update e values
        self.e_values = self.g * self.l * self.e_values + rewards
        
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
    
    def save_model(self, directory):
        #store model
        self.network.save_model(directory=directory)
    
    
    