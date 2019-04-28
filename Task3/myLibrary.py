# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:03:08 2019

@author: emile
"""
import numpy as np

def CreateApplesMatrix(apples, myLoc, myOr):
    a = np.zeros([15,15])
    for apple in apples:
        #center around location ([7,7] is location of player)
        apple = apple - myLoc + [7,7]
        #[7,7] or [8,8] ?
        if min(apple) < 0 : print("ERROR: Translation not sufficient")
        a[apple[0],apple[1]] = 1 #more elegant way of doing this?
    return rotate(a, myOr)


#TODO possibility to make use of limited total number of players?
def createPlayerMatrix(players, playing, myLoc, myOrientation):
    a = np.zeros([15,15])
    for player in players:
        loc = player["location"]
        loc = loc - myLoc + [7,7]
        a[loc[0],loc[1]] = 1
    return rotate(a, myOrientation)

def rotate(a, myOr):
    #standard orientation up
    if myOr == "right":
        a = np.rot90(a)
    elif myOr == "left":
        a = np.rot90(a,3)
    elif myOr == "down":
        a = np.rot90(a,2)
    elif myOr == "up": 
        pass
    else : print("ERROR: orientation is right up left or down")
    return a