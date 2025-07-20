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
import importlib
import trainRipNet
importlib.reload(trainRipNet)
from trainRipNet import train_model
from utils import set_seed

def main_train():
    """
    Main training script for Ripple detection using RipNetCNN.
    Build naive model, create dataloaders, get pos_weight (ripples are sparse), set training parameters, run training loop
    """
    #build dataset and naive model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42) #for reproducibility
    dataset = RippleDataset(data_path='data/', annotations_path='annotations/')
    model = RipNetCNN()
    
    #split dataset into training (70%), validation (15%)  and testing (15%) datasets
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    #set seed via generator to get same examples in each group every time
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    #Build dataloaders for batching data
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) 
    
    ## Get pos_weight to rebalance, as ripples are sparse
    #Concatenate all labels
    all_labels = []
    for _, labels in train_loader: #only training data, otherwise leaking
        all_labels.append(labels)
    all_labels_tensor = torch.cat(all_labels, dim=0) # Concatenate to a single tensor
    
    # Count positive and negative samples
    num_positives = (all_labels_tensor == 1).sum().item()
    num_negatives = (all_labels_tensor == 0).sum().item()  
    # Compute ratio
    r = num_negatives / (num_positives + 1e-8)  # avoid div by zero

    pos_weight_tensor = torch.tensor([r], device=device)
    
    train_model(model, train_loader, val_loader, test_loader, pos_weight_tensor,threshold=0.7, epochs=20, lr=1e-4)
