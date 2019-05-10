# -*- coding: utf-8 -*-
"""
Created on Thu May  2 00:22:55 2019

@author: emile
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


#TODO generate random training data
x_train = []


model = Sequential()
model.add(Dense(430,activation='relu',input_dim=430))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(13,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(430,activation='relu'))

model.compile(loss=keras.losses.mean_squared_error,
             optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0),
             metrics = ['accuracy'])

model.fit(x_train,x_train,verbose=1,epochs=10,batch_size=128)
model.save('TestModel.h5')
#del model