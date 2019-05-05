# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:12:26 2019

@author:
    
    
An abstract base class agent for the Apples Game
"""
import numpy as np
from abc import ABC, abstractmethod

class agent_abstract(ABC):
    def __init__(self, network, player, nb_rows, nb_cols, nb_players, state_size, action_size):
        self.player = player
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        self.action_size = action_size
        self.state_size = state_size
        
        self.network = network
        
        #retain last act information to push to memory when result of action is observed
        # When dirty_bit is True, the agent must first receive an observation before acting again
        self.last_action = 0
        self.last_state = np.zeros((15,15))
        self.dirty_bit = False
        
        
    
    
    def next_action(self, players, apples):
        state = self.calc_state(players,apples)
        
        #Feed to network
        q_values = self.network.act(state = state)
        action_index = self.q_values_to_action(q_values)
        
        #Store action and state and set dirty bit
        if self.dirty_bit == False:
            self.last_action = action_index
            self.last_state = state
            self.dirty_bit = True
        else:
            print("\n\nERROR: dirty_bit is True! An observation is expected before a new action request.\n\n")
            return -1
        
        return action_index
    
    
    
    
    def observe(self, rewards, players, apples, done):
        next_state = self.calc_state(players,apples)
        #subjective_reward = self.calc_subjective_reward(rewards)
        subjective_reward = rewards[self.player - 1]
        
        #new_DQNAgent
        #self.network.observe(reward=subj_reward, terminal=done)
        
        #Keras_DQNAgent
        if self.dirty_bit == True:
            self.network.remember(state=self.last_state,action=self.last_action,reward=subjective_reward,next_state=next_state,done=done)
            self.dirty_bit = False
        else:
            print("\n\nERROR: dirty_bit is False! An action is expected before a new observation can be made.\n\n")  
    
    
    
    
    def calc_state(self, players, apples):
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
            
        return state
    
    
    
    @abstractmethod
    def calc_subjective_reward(self, rewards):
        pass

    @abstractmethod
    def q_values_to_action(self, q_values):
        pass