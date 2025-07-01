# -*- coding: utf-8 -*-
"""
Created on Sep 8, 2023.

@author: Claire Cooper
"""

import numpy as np
import os
import scipy.io
from butter import lowpass
import re 


def dataset_generator(data_path, annotations_path, fs):
    """Create the train and test (validation) dataset."""
    first_round = True
    binary_traces = []
    for name_data in os.listdir(data_path):

        # get the recording
        data = scipy.io.loadmat(data_path + name_data)
        key = list(data.keys())[-1]
        data = data[key]

        # set the channels to be the first dimension
        if data.shape[1] < data.shape[0]:
            data = data.T

        # select the last eight channels
        data = data[:8, :]
        FPs=False
        found_labels = False
        # get the labels corresponding to the loaded recording
        for name_labels in os.listdir(annotations_path):
            if name_labels[-3:] != 'mat':
                continue
            if name_labels.split('.')[1] == name_data[7:].split('.')[0]:
                print('Creating dataset from {:}'.format(name_data))
                found_labels = True
                keyname=list(scipy.io.loadmat(annotations_path + name_labels).keys())[-1]
                timestamps = scipy.io.loadmat(
                    annotations_path + name_labels)[keyname][0][0][0]
                FPfiles=os.listdir(annotations_path+'FPs/')
                if len(FPfiles ) > 0:
                    for file in range(len(FPfiles)):
                        if FPfiles[file].__contains__(keyname):
                            FPTimes=scipy.io.loadmat(
                                annotations_path + 'FPs/'+FPfiles[file])['FPs'][0][0][0]
                            FPs=True
                else:
                    FPTimes=0
                    FPs=False
        if found_labels:
            # z_score the data
            z_scored_signals = np.zeros((data.shape))

            for ch in range(data.shape[0]):
                z_scored_signals[ch, :] = (data[ch, :] - np.mean(data[ch, :])
                                           ) / np.std(data[ch, :])

            # low pass filder the signals
            z_scored_signals = lowpass(z_scored_signals, fs //4, fs)

            # generate positive labels
            half_win = fs // 6  
            full_win = fs // 6 * 2  
            x_pos_pre = []
            y_pos_pre = []
            binary_trace = np.zeros((data.shape[1]))
            for t in timestamps:
                start_dp = int(round(t[0] * fs))
                stop_dp = int(round(t[1] * fs))
                ripple_length = stop_dp - start_dp
                if (ripple_length <= 0) or (stop_dp > data.shape[1]):
                    continue
                ripple_center = start_dp + ripple_length // 2
                binary_trace[start_dp:stop_dp] = np.linspace(
                    1, 1, ripple_length)
                for _ in range(5):
                    shift = np.random.randint(400) - 200
                    current_x = z_scored_signals[
                        :, ripple_center - half_win + shift:
                            ripple_center + half_win + shift]
                    if current_x.shape[1] == full_win:
                        x_pos_pre.append(current_x)
                        y_pos_pre.append(binary_trace[
                            ripple_center - half_win + shift:
                                ripple_center + half_win + shift])

            x_pos_pre = np.array(x_pos_pre)
            y_pos_pre = np.array(y_pos_pre)

            # generate negative labels
            x_neg_pre = []
            y_neg_pre = []
            last_annotation_dp = timestamps[-1][-1] * fs
            neg_count = 0
            
            if FPs==True:
                numNegs=x_pos_pre.shape[0]-len(FPTimes)
            else:
                numNegs=x_pos_pre.shape[0]
            while neg_count < numNegs:
                random_dp = np.random.randint(last_annotation_dp)
                test_trace = binary_trace[random_dp - half_win:
                                          random_dp + half_win]
                if np.sum(test_trace) == 0:
                    current_x = z_scored_signals[:, random_dp - half_win:
                                                 random_dp + half_win]
                    if current_x.shape[1] == full_win:
                        x_neg_pre.append(current_x)
                        y_neg_pre.append(test_trace)
                        neg_count += 1
            if FPs== True:
                for fp in range(len(FPTimes)):
                    curx= z_scored_signals[:,int(FPTimes[fp]*fs)-half_win:int(FPTimes[fp]*fs)+half_win]
                    x_neg_pre.append(curx)
                    y_neg_pre.append(np.zeros(full_win,))
                    
            x_neg_pre = np.array(x_neg_pre)
            y_neg_pre = np.array(y_neg_pre)

          
            x_data=np.concatenate((x_pos_pre,x_neg_pre),axis=0)
            y_label=np.concatenate((y_pos_pre,y_neg_pre),axis=0)
           
            if first_round:
                dataX = x_data
                labelsY = y_label
                first_round = False
            else:
                dataX = np.concatenate((dataX, x_data),axis=0)
                labelsY = np.concatenate((labelsY, y_label),axis=0)
               

            binary_traces.append(binary_trace)

    return (dataX,labelsY, binary_traces)
