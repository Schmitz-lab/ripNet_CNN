#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:10:34 2025

@author: claire
"""

import torch
from torch.utils.data import DataLoader, random_split
from ripDataset import RippleDataset
from RipNet import RipNetCNN
from utils import *
from trainRipNet import train_model

#build dataset and naive model
dataset = RippleDataset(data_path='data/', annotations_path='annotations/')
model = RipNetCNN()

#split dataset into training (70%), testing (15%)  and validation (15%) datasets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#get pos_weight to rebalance, as ripples are sparse
all_labels = []
for _, labels in train_loader:
    all_labels.append(labels)

# Concatenate to a single tensor
all_labels_tensor = torch.cat(all_labels, dim=0)

# Count positive and negative samples
num_positives = (all_labels_tensor == 1).sum().item()
num_negatives = (all_labels_tensor == 0).sum().item()

# Compute ratio
r = num_negatives / (num_positives + 1e-8)  # avoid div by zero

print(f"pos_weight: {r:.2f}")
device = "cuda"
pos_weight_tensor = torch.tensor([r], device=device)

train_model(model, train_loader, val_loader, test_loader, pos_weight_tensor, epochs=20, lr=1e-3)
