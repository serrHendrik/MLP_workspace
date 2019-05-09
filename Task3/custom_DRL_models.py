# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:02:52 2019

This file defines our custom Deep Reinforcement Learning agents, 
build on top of the TensorForce framework.

"""
from tensorforce.agents import DQNAgent
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop 


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




def default_model(input_shape=(15,15,1), action_size=4):    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model




def cnn_model_v1(input_shape=(15,15,1), action_size=4):
    model = Sequential()
    model.add(Conv2D(32,
                      3,
                      strides=(2, 2),
                      padding="valid",
                      activation="relu",
                      input_shape=input_shape,  
                      data_format="channels_last"))
    model.add(Conv2D(64,
                      3,
                      strides=(1, 1),
                      padding="valid",
                      activation="relu"))
    model.add(Conv2D(64,
                      3,
                      strides=(1, 1),
                      padding="valid",
                      activation="relu"))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(action_size))
    model.compile(loss="mean_squared_error",
                   optimizer=RMSprop(lr=0.00025,
                                     rho=0.95,
                                     epsilon=0.01),
                   metrics=["accuracy"])
    return model

def cnn_model_v2(input_shape=(15,15,1), action_size=4):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(action_size))
    
    # initiate RMSprop optimizer
    opt = RMSprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
