#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 09:32:22 2025

@author: claire
"""

import torch
from RipNet import RipNetCNN
from gradcam_utils import generate_gradcam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
Run gradCAM
Load trained model, load saved inputs, labels and outputs from pickled file (created with save_a_batch.py)

"""

#Variables
fs=1250
model_path='best_model_pytorch/best_model.pth'
pkl_path='model_output_pkl/batch_output.pkl'

#Load model and pick target layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RipNetCNN().to(device)
model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))  
model.eval()
target_layer = model.features[0] #0 is first, 9 is last, access with for idx, layer in enumerate(model.features):

#open saved model inputs and outputs 
with open("/alznew/claire/ripples_in_vivo_from_Claire/pytorchModel/model_output_pkl/batch_output.pkl", "rb") as f:
    data = pickle.load(f)
inputs = data["inputs"]
labels = data["labels"]
preds = data["preds"]

#Loop through examples to plot multiple heatmaps in one go
for example_idx in np.arange(63,128,2):
    
    input_tensor = torch.tensor(inputs[example_idx], dtype=torch.float32).unsqueeze(0) # [1, 1, 8, 416]
    cam = generate_gradcam(model, input_tensor, device, target_layer)
    
    #Plot gradcam
    input_data = inputs[example_idx].squeeze() # shape: [8, 416]
    time = np.arange(input_data.shape[1]) / fs * 1000  # time in ms
    
    # input_data: [channels, time]
    # cam: [channels, time]
    offset = 3  # vertical spacing between channels
    plt.figure(figsize=(10, 6))
    threshold = 0.7
    
    #Plot model prediction (blue) and ground truth (green)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (8,8))  # 2 rows, 1 column, share x-axis
    ax1.plot(time,preds[example_idx], label='Raw Predictions', alpha=0.5)
    ax1.plot(time,labels[example_idx], label='Ground Truth', color='green', alpha=0.5)
    ax1.axhline(threshold, color='gray', linestyle='--', label=f'Threshold={threshold}')
    ax1.set_ylim(-0.1,1)
    ax1.set_ylabel('Probability')         

    ax1.set_title("Ground truth vs. Predictions")
    handles, titles = ax1.get_legend_handles_labels()
    unique = dict(zip(titles, handles))
    ax1.legend(unique.values(), unique.keys())
    
    # Show Grad-CAM as background in second plot below
    im=ax2.imshow(cam, aspect='auto', extent=[time[0], time[-1], 0, 8 * offset],
               origin='lower', cmap='hot', alpha=0.4)
    cax = inset_axes(ax2, width="2.25%", height="100%", loc='upper left',
                 bbox_to_anchor=(1.01, 0., 1, 1),
                 bbox_transform=ax2.transAxes,
                 borderpad=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Grad-CAM Intensity')
    # Overlay LFP traces
    for ch in range(input_data.shape[0]):
        ax2.plot(time, input_data[ch] + ch * offset, color='black', linewidth=1)
        
    ax2.tick_params(labelleft=False) 
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Channels")
    ax2.set_title("Input signal with Grad-CAM heatmap")
    fig.tight_layout()
    plt.show(fig)
    plt.close(fig)

