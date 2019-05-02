# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:02:52 2019

This file defines our custom Deep Reinforcement Learning agents, 
build on top of the TensorForce framework.

"""
from tensorforce.agents import DQNAgent


# TODO: Add preprocessing (gray scale) + greedy-epsilon exploration and an actual network
def new_DQNAgent(nb_actions):
    #batch_size = 4096
    states_spec = dict(shape=(15,15), type='float')
    actions_spec = dict(num_actions=nb_actions, type='int')
    network_spec = [
        dict(type='flatten'),
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ]
        
    agent = DQNAgent(
        states=states_spec,
        actions=actions_spec,
        network=network_spec,
        actions_exploration = dict(
            type='epsilon_decay'
        ),
        target_sync_frequency=50,
        double_q_model=True
    )
    
    agent.reset()
    return agent


