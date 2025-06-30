#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:06:58 2025

@author: claire
"""

import torch
import torch.nn as nn

class RipNetCNN(nn.Module):
    def __init__(self, input_shape = (1,8,416)):
        super().__init__()
                
        self.features = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=1, stride=2, padding=0),  # same padding for 1x1 is 0
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Dropout(0.25)
        )

    # Compute output size after conv layers (optional: dynamic way shown below)
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            out = self.features(dummy_input)
        self.flatten_dim = out.view(1, -1).shape[1]
    
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 416),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x