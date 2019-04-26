# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:44:02 2019

@author: serru
"""
import numpy as np
import random


class fictitiousPlayerRPS:
    
    #initialBeliefs = [<number of Rocks played>, <number of Paper played>, <number of Scissors played>]
    def __init__(self, initialBeliefs = [1,1,1]):
        self.beliefs = np.array(initialBeliefs)
        self.beliefs_timeline = np.array([self.beliefs / float(sum(self.beliefs))])
        
        if initialBeliefs[0] == initialBeliefs[1] and initialBeliefs[0] == initialBeliefs[2]:
            self.expected = random.randint(0,2)
        else:
            self.expected = initialBeliefs.index(max(initialBeliefs))
            
        #Track total earned reward
        self.total_reward = 0
            
    def play(self):
        return self.bestResponse(self.expected)
        
    #deduces play of adversary, and updates beliefs + expected play
    def update(self, played, opponentPlay, payoff):
        """
        if payoff == -1 : opponentPlay = bestResponse(played) #deduce the play of adversary
        elif payoff == 0 : opponentPlay = played
        elif payoff == 1 : opponentPlay = self.expected
        else: 
            print("ERROR: payoff is -1,0 or 1")
            return -1
        """
        
        self.total_reward += payoff
        
        self.beliefs[opponentPlay] += 1
        #print("I played " + str(played) + ". Payoff: " + str(payoff) + ". Beliefs: " + str(self.beliefs))
        self.beliefs_timeline = np.append(self.beliefs_timeline,[np.copy(self.beliefs) / float(sum(self.beliefs))],axis=0)
        if self.beliefs[opponentPlay] > self.beliefs[self.expected]:
            self.expected = opponentPlay
        elif self.beliefs[opponentPlay] == self.beliefs[self.expected]:
            if random.random() > 0.5 : self.expected = opponentPlay
    
        #Play CODES:    0 : Rock
    #               1 : Paper
    #               2 : Scissors
    def bestResponse(self, play):
        if play == 0 : return 1
        elif play == 1 : return 2
        elif play == 2 : return 0
        else :
            print("ERROR: play has to be 0,1 or 2")
            return -1