#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:06:58 2025

@author: claire
"""

import torch
import torch.nn as nn

class RipNetCNN(nn.Module):
    """
 CNN model for ripple detection with dynamically calculated classifier input size.
 Expects input of shape (batch_size, 1, 8, 416).
 """
    def __init__(self, input_shape = (1,8,416)): # (C,H,W), shape of single example (w/o batch size)
        super().__init__() #inherit capabilities of nn.Module, e.g. model.eval()
                
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1)), #convolve over time dimension
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(8,1), stride=2, padding=0), #convolve over channels dimension 
            nn.BatchNorm2d(32),
            nn.ReLU(),
        
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),      
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Dropout(0.4)
            )

    # Compute output size after conv layers 
        dummy_input = torch.zeros(1, *input_shape) # * unpacks tuple so elements passed separately
        with torch.no_grad():
            out = self.features(dummy_input)
        self.flatten_dim = out.view(1, -1).shape[1] #.view(1,-1) reshape tensor to keep batch size and infer second element, then take second element as flat_dim
    
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 416),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten output from conv layers
        x = self.classifier(x)
        return x