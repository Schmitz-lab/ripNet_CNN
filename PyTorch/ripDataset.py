#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:16:04 2025

@author: claire
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import scipy.io
from butter import lowpass

class RippleDataset(Dataset):
    def __init__(self, data_path, annotations_path, fs=1250, chunk_len=416, shift_range=200):
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.fs = fs
        self.chunk_len = chunk_len
        self.shift_range = shift_range
        
        self.X = []  # to store data chunks
        self.Y = []  # to store binary labels
        
        self.prepare_dataset()
        
    def prepare_dataset(self):
        half_win = self.chunk_len // 2
        full_win = self.chunk_len

        for name_data in os.listdir(self.data_path):
            if not name_data.endswith('.mat'):
                continue

            data_mat = scipy.io.loadmat(os.path.join(self.data_path, name_data))
            key = list(data_mat.keys())[-1]
            data = data_mat[key]

            if data.shape[1] < data.shape[0]:
                data = data.T

            # Match annotation file
            found_labels = False
            for name_labels in os.listdir(self.annotations_path):
                if not name_labels.endswith('.mat'):
                    continue
                if name_labels.split('.')[1] == name_data[7:].split('.')[0]:
                    found_labels = True
                    label_mat = scipy.io.loadmat(os.path.join(self.annotations_path, name_labels))
                    keyname = list(label_mat.keys())[-1]
                    timestamps = label_mat[keyname][0][0][0]
                    break  # Stop after finding the matching label file

            if not found_labels:
                continue

            # Optional: Load false positives if folder exists
            FPs = False
            FPTimes = []
            fp_folder = os.path.join(self.annotations_path, 'FPs')
            if os.path.exists(fp_folder):
                for fp_file in os.listdir(fp_folder):
                    if keyname in fp_file:
                        FPTimes = scipy.io.loadmat(os.path.join(fp_folder, fp_file))['FPs'][0][0][0]
                        FPs = True

            # Z-score and filter
            z_scored = np.zeros_like(data)
            for ch in range(data.shape[0]):
                z_scored[ch] = (data[ch] - np.mean(data[ch])) / np.std(data[ch])
            z_scored = lowpass(z_scored, self.fs // 4, self.fs)

            binary_trace = np.zeros(data.shape[1])
            x_pos, y_pos = [], []

            # Generate positive samples
            for t in timestamps:
                start_dp = int(round(t[0] * self.fs))
                stop_dp = int(round(t[1] * self.fs))
                if stop_dp > data.shape[1] or stop_dp <= start_dp:
                    continue
                center = (start_dp + stop_dp) // 2
                binary_trace[start_dp:stop_dp] = 1.0

                for _ in range(5):  # augment with shifts
                    shift = np.random.randint(-self.shift_range, self.shift_range)
                    chunk_start = center - half_win + shift
                    chunk_stop = center + half_win + shift
                    if chunk_start < 0 or chunk_stop > data.shape[1]:
                        continue
                    chunk = z_scored[:, chunk_start:chunk_stop]
                    if chunk.shape[1] == full_win:
                        x_pos.append(chunk)
                        y_pos.append(binary_trace[chunk_start:chunk_stop])

            # Generate negative samples
            x_neg, y_neg = [], []
            last_dp = int(timestamps[-1][-1] * self.fs)
            num_negs = len(x_pos) - len(FPTimes) if FPs else len(x_pos)
            neg_count = 0
            while neg_count < num_negs:
                rand_dp = np.random.randint(half_win, last_dp - half_win)
                if np.sum(binary_trace[rand_dp - half_win:rand_dp + half_win]) == 0:
                    chunk = z_scored[:, rand_dp - half_win:rand_dp + half_win]
                    if chunk.shape[1] == full_win:
                        x_neg.append(chunk)
                        y_neg.append(np.zeros(full_win))
                        neg_count += 1

            if FPs:
                for fp in FPTimes:
                    fp_dp = int(fp * self.fs)
                    if fp_dp - half_win < 0 or fp_dp + half_win > data.shape[1]:
                        continue
                    chunk = z_scored[:, fp_dp - half_win:fp_dp + half_win]
                    if chunk.shape[1] == full_win:
                        x_neg.append(chunk)
                        y_neg.append(np.zeros(full_win))

            # Combine pos and neg
            X_all = np.concatenate([x_pos, x_neg], axis=0)
            Y_all = np.concatenate([y_pos, y_neg], axis=0)

            self.X.extend(X_all)
            self.Y.extend(Y_all)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y