# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:02:52 2019

This file defines our custom Deep Reinforcement Learning agents, 
build on top of the Keras framework.

"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop 



def default_model_v1(input_shape=(15,15,1), action_size=4):    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

def default_model_v2(input_shape=(15,15,1), action_size=4):    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
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
