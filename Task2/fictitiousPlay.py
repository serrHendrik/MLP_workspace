# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:42:03 2019

@author: emile

Fictitious play
Based on book Multiagent Systems: Algorithmic, Game-Theoretic, and logical functions (Shoham)
"""

import numpy as np
import random

#Play CODES:    0 : Rock
#               1 : Paper
#               2 : Scissors
def bestResponse(play):
    if play == 0 : return 1
    elif play == 1 : return 2
    elif play == 2 : return 0
    else :
        print("ERROR: play has to be 0,1 or 2")
        return -1


class fictitiousPlayerRPS:
    
    #initialBeliefs = [<number of Rocks played>, <number of Paper played>, <number of Scissors played>]
    def __init__(self, initialBeliefs = [1,1,1]):
        self.beliefs = initialBeliefs
        self.expected = initialBeliefs.index(max(initialBeliefs))
    
    def play(self):
        return bestResponse(self.expected)
        
    #deduces play of adversary, and updates beliefs + expected play
    def update(self, played, payoff):
        if payoff == -1 : opponentPlay = bestResponse(played) #deduce the play of adversary
        elif payoff == 0 : opponentPlay = played
        elif payoff == 1 : opponentPlay = self.expected
        else: 
            print("ERROR: payoff is -1,0 or 1")
            return -1
        self.beliefs[opponentPlay] += 1
        if self.beliefs[opponentPlay] > self.beliefs[self.expected]:
            self.expected = opponentPlay
        elif self.beliefs[opponentPlay] == self.beliefs[self.expected]:
            if random.random() > 0.5 : self.expected = opponentPlay
        
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
            self.p1.update(play_p1,payoff_p1)
            self.p2.update(play_p2,payoff_p2)
            self.results[play_p1, play_p2] += 1
        return self.results

#p1 = fictitiousPlayerRPS([12,5,3])
#p2 = fictitiousPlayerRPS([6,21,9])
#game = RPSgame(p1,p2) 
game = RPSgame()
results = game.play()
print(results)

#Findings: fictitious play very influencable by initial beliefs
#What when beliefs of 2 actions are equal? (now just maintain same strategy)       
