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
#AGENT_TYPE = "IA"
MODEL_TYPE = "DDQN"



files = {
        "DQN": ["models/DQN_MODEL_loss_per_minibatch.csv", "models/DQN_MODEL_loss_total.csv"],
        "DDQN": ["models/DDQN_MODEL_4actions_IA_loss_per_minibatch.csv", "models/DDQN_MODEL_4actions_IA_loss_total.csv"]
    }


def read_file(filename):
    with open(filename, 'r') as rf:
        reader = csv.reader(rf)
        loss = np.array(list(reader))
    rf.close()
    loss = loss[1:,:].flatten().astype(np.float64)    
    return loss

def display_loss(MODEL_TYPE):
   #Display IA Loss
    loss_mb = read_file(files[MODEL_TYPE][0])
    loss_total = read_file(files[MODEL_TYPE][1])
    title = MODEL_TYPE + " Loss function" 
    
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
display_loss(MODEL_TYPE)

