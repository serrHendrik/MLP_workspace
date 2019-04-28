# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:53:03 2019

@author: emile
"""

import json

x = {
  "name": "John",
  "age": 30,
  "married": True,
  "divorced": False,
  "children": ("Ann","Billy"),
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

yy = json.loads(y)
print(yy["name"])
print(yy["cars"][0])
cars = yy["cars"]
cars[0]["model"]

#%%

import numpy as np
msg =  json.dumps({
        "type": "start",
        "player": 1,
        "game": "123456",
        "grid": [36, 16],
        "players": [
          {
            "location": [7, 10],
            "orientation": "right"
          },
          {
            "location": ["?", "?"],
            "orientation": "?"
          }
        ],
        "apples": [[10, 15], [3, 5]]
    })
        

msg = json.loads(msg)
players = msg["players"]
playing = msg["player"]
myLoc = players[playing-1]["location"]
myOrientation = players[playing-1]["orientation"]