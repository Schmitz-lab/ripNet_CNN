#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 1, 2023

@author: Claire Cooper
"""


from tensorflow.keras.layers import (Conv2D,Flatten, Dense, Dropout,BatchNormalization)
from tensorflow.keras.models import Sequential


class RipNetClaire:
    def build(width,height,depth,reg,init="he_normal"):
        model=Sequential()
        inputShape=(height,width,depth)
        chanDim=-1
        
        model.add(Conv2D(16,(1,1),strides=(2,2),padding="same",activation="relu",kernel_regularizer=reg,input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),strides=(2,2),padding="same",activation="relu",kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),strides=(2,2),padding="same",activation="relu",kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256,(3,3),strides=(2,2),padding="same",activation="relu",kernel_regularizer=reg))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(416,activation="sigmoid"))
        return model
        

