#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 1, 2023

@author: Claire Cooper
"""


from RipNetClaire import RipNetClaire
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from dataset_generation import dataset_generator


def model_training(epochs):
    """Prepare the data, train and test the model on the validation dataset."""
    root = os.getcwd()
    data_path = root + '/data/'
    annotations_path = root + '/annotations/'
    fs = 1250 

    # create the dataset
    output_data = dataset_generator(data_path, annotations_path, fs)
    data=np.expand_dims(output_data[0],-1)
    labels=output_data[1]
    labelvals=np.ones(labels.shape[0])
    labelvals[np.sum(labels,axis=1)==0]=0
    (trainX,testX,trainY,testY)=train_test_split(data,labels,stratify=labelvals)

    #create instance of model
    print("compiling model...")
    opt=Adam(lr=1e-4,decay=1e-4/epochs)
    model=RipNetClaire.build(width=416,height=8,depth=1,reg=l2(0.001))
    model.compile(loss='mean_squared_error',
                  optimizer=opt,metrics=["mean_squared_error"])
    print("Training model for "+ str(epochs)+ " epochs...")
    
    # define the callbacks
    root = os.getcwd()
    checkpoint_path = root + '/trained_model/model.h5'
    patience_epochs = 4
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience_epochs)]
    
    #train the model
    r=model.fit(x=trainX,y=trainY,batch_size=256,validation_data=(testX,testY),callbacks=callbacks,epochs=epochs)
    
    print('Val loss: ', np.min(r.history['val_loss']))

    # plot the loss
    plt.figure()
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.plot(r.history['mean_squared_error'],label='mean_squared_error')
    plt.legend()
    plt.show()

    # load the best model (weights)
    model.load_weights(checkpoint_path)


    pred=model.predict(testX)
    testX=np.squeeze(testX)
 
    # plot example predictions
    #rip=11
    #plt.figure()
    #for chan in range(testX.shape[1]):
     #   plt.plot(testX[rip,chan,:]+chan+3+(2*chan))
    #plt.plot(testY[rip,:],label='ground truth')
    #plt.plot(pred[rip],label='model prediction')
    #plt.legend()
    #plt.show() 
    

    return model.summary()
