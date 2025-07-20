#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:52:39 2025

@author: claire
"""

import torch
from RipNet import RipNetCNN
from utils import set_seed
from ripDataset import RippleDataset
import matplotlib.pyplot as plt
import numpy as np

def compute_saliency_ripnet(model, lfp_window):
    """
    Compute a saliency map for a given model input (2d chunk)
    
    lfp_window: numpy array of shape [8, 416]
   
    """
    model.eval()
    
    # Prepare input
    input_tensor = torch.tensor(lfp_window, dtype=torch.float32).unsqueeze(0)  # [1, 1, 8, 416]
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)  # shape: [1, 416]
    probs = torch.sigmoid(output)
    
    # Select the timepoint prediction to explain
    # get the max prob timepoint...during highest prob/first highest prob ripple for pos examples
    target_time_idx = probs[0].argmax().item() 
    score = output[0, target_time_idx]
    
    # Backward pass
    model.zero_grad()
    score.backward() #gradient at ripple time w.r.t. input

    # Get saliency: gradient of score w.r.t. input
    saliency = input_tensor.grad.abs().squeeze().detach().cpu().numpy()  # shape: [8, 416]

    # Plot
    # Plot: overlay raw traces + saliency
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot saliency heatmap (in background)
    ax.imshow(
        saliency,
        aspect='auto',
        cmap='seismic', #hot
        alpha=0.4,
        extent=[0, 416, 0, 8] , # x: time, y: channels
        origin='lower'
    )

    # Plot raw LFP traces (with vertical offsets)
    offset = 2.5  # spacing between traces
    time = np.arange(416)
    
# Convert lfp_window to numpy if it's a tensor
    if isinstance(lfp_window, torch.Tensor):
        lfp_window = lfp_window.squeeze(0).detach().cpu().numpy()
    print(lfp_window.shape)
    
    fig, ax = plt.subplots(figsize=(9,6))

# Plot heatmap on ax
    im = ax.imshow(
        saliency,
        aspect='auto',
        cmap='hot',
        alpha=0.3,
        extent=[0, 416, 0, 8],
        origin='lower'
    )
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Channels (heatmap)')
    
    # Create second y-axis for LFP traces
    ax2 = ax.twinx()
    
    # Plot LFP traces with offset on ax2
    for ch in range(8):
        ax2.plot(time, lfp_window[ch] + ch * offset, color='black', linewidth=0.8)
    
    ax2.set_ylim(-offset, 8*offset)
    ax2.set_ylabel('LFP traces (offset)')
    
    # Hide y ticks on ax2 
    ax2.set_yticks([])
    
    plt.title('Saliency heatmap + LFP traces')
    plt.show()

    return saliency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42) #for reproducibility
dataset = RippleDataset(data_path='data/', annotations_path='annotations/')
model = RipNetCNN()
model_dict =torch.load('best_model_pytorch/best_model.pth', weights_only=True)
model.load_state_dict(model_dict)
X,Y = dataset[3] #ex 5 has a ripple

compute_saliency_ripnet(model, X)