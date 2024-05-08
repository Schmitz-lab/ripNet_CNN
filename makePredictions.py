#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Sep 5, 2023

@author: Claire Cooper
"""
from RipNetClaire import RipNetClaire
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scipy.io
from butter import lowpass
from scipy.signal import find_peaks


def makePredictions(model_path,test_path,thr):
#Load trained model

    model=RipNetClaire.build(width=416,height=8,depth=1,reg=l2(0.001))
    model.load_weights(model_path+'model8chans.h5')
    
    predFolder='modelPreds/'
    
    for name_data in os.listdir(test_path):
        if name_data[-3:] != 'mat':
            continue
        # get the recording
        data = scipy.io.loadmat('testing_data/' + name_data)
        key = list(data.keys())[-1]
        data = data[key]
        
         # set the channels to be the first dimnetion
        if data.shape[1] < data.shape[0]:
            data = data.T
        
         # select the last four channels
        data = data[:8, :]
        #z-score the signal
        z_scored_signals = np.zeros((data.shape))
        fs=1250
        for ch in range(data.shape[0]):
            z_scored_signals[ch, :] = (data[ch, :] - np.mean(data[ch, :])) / np.std(data[ch, :])
        # low pass filder the signals
        z_scored_signals = lowpass(z_scored_signals, fs //4, fs)
    
        # get some parameters
        full_win = fs // 6 * 2
        # generate the prediction_trace
        border_left = 0
        channels_numb = z_scored_signals.shape[0]
        input_data = np.zeros((int(z_scored_signals.shape[1] / full_win),
                           channels_numb, full_win))
    
        for win in range(int(z_scored_signals.shape[1] / full_win) + 1):
            border_right = border_left + full_win
            if border_right > z_scored_signals.shape[1]:
                break
            data_win = np.zeros((channels_numb, full_win))
            for ch in range(channels_numb):
                data_win[ch, :] = z_scored_signals[ch, border_left:border_right]
            input_data[win] = data_win
            border_left = border_left + full_win
    
        input_data=np.expand_dims(input_data.reshape(-1,channels_numb, full_win),-1)
        pred = model.predict(input_data)
        prediction_trace = pred.reshape(-1,)
        prediction_trace = lowpass(prediction_trace, fs // 20, fs)
    
        # get the ripples from the prediction_trace
        thr1 = thr # 0.20  # arbitrary chosen
        x_dp = []
        y = []
    
        for seg in range(prediction_trace.shape[0] // full_win):
        
            lim_left = seg * full_win
            lim_right = min(lim_left + full_win, prediction_trace.shape[0])
            current_pred = prediction_trace[lim_left:lim_right]
        
            # if in the segment current_pred the max value is above the threshold
            if np.max(current_pred) >= thr1:
        
                # search for peaks
                peaks = find_peaks(current_pred.reshape(-1, ))
        
                # iterate through the peaks
                for px in peaks[0]:
        
                    # if the conditions below are met, append x_dp and y
                    if len(x_dp) > 0:
                        max_valley = y[-1] * 0.8  # arbitrary chosen
                        last_x_dp = x_dp[-1]
                    else:
                        max_valley = current_pred[px] * 0.8  # arbitrary chosen
                        last_x_dp = 0
                    if np.min(prediction_trace[last_x_dp:lim_left + px]
                              ) < max_valley:
                        if current_pred[px] > thr1:
                            x_dp.append(lim_left + px)
                            y.append(prediction_trace[lim_left + px])
                    else:
                        if (len(y) > 0) and (current_pred[px] > y[-1]):
                            y = y[:-1]
                            x_dp = x_dp[:-1]
                            x_dp.append(lim_left + px)
                            y.append(prediction_trace[lim_left + px])
        np.savetxt(predFolder + name_data[7:-4] + '_predTrace.txt',prediction_trace) 
        np.savetxt(predFolder + name_data[7:-4]+ '_ripTimes.txt',x_dp)                 
