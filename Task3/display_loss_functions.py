# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:03:39 2019

@author:
    
Simple script to quickly visualize loss functions

"""
import csv
import matplotlib.pyplot as plt
import numpy as np

#What do you want to see?
AGENT_TYPE = "FR"
MODEL_TYPE = "DDQN"



files = {
    "IA": {
        "DQN": ["models/DQN_IA_MODEL_loss_per_minibatch.csv", "models/DQN_IA_MODEL_loss_total.csv"],
        "DDQN": ["models/DDQN_IA_MODEL_loss_per_minibatch.csv", "models/DDQN_IA_MODEL_loss_total.csv"]
        },
    
    "FR": {
        "DQN": ["models/DQN_FR_MODEL_loss_per_minibatch.csv", "models/DQN_FR_MODEL_loss_total.csv"],
        "DDQN": ["models/DDQN_FR_MODEL_loss_per_minibatch.csv", "models/DDQN_FR_MODEL_loss_total.csv"]
        }
    }


def read_file(filename):
    with open(filename, 'r') as rf:
        reader = csv.reader(rf)
        loss = np.array(list(reader))
    rf.close()
    loss = loss[1:,:].flatten().astype(np.float64)    
    return loss

def display_loss(AGENT_TYPE, MODEL_TYPE):
   #Display IA Loss
    loss_mb = read_file(files[AGENT_TYPE][MODEL_TYPE][0])
    loss_total = read_file(files[AGENT_TYPE][MODEL_TYPE][1])
    title = MODEL_TYPE + " Loss function for " + AGENT_TYPE + " agents"
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    f.suptitle(title)
    ax1.plot(loss_mb)
    ax1.set_title("Per minibatch")
    ax1.set_xlabel("Minibatches")
    ax1.set_ylabel("Loss")
    ax2.plot(loss_total)
    ax2.set_title("Per episode")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Loss")
    plt.yscale('log')
    plt.show()
    
    if loss_total.size >= 10:
        print("Loss values for last 10 episodes:")
        print(str(loss_total[-10:])) 


#Display loss
display_loss(AGENT_TYPE, MODEL_TYPE)

