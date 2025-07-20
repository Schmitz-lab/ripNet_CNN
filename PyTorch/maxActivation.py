#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:49:47 2025

@author: claire
"""

import matplotlib.pyplot as plt
import torch
from RipNet import RipNetCNN
from utils import set_seed
from ripDataset import RippleDataset

"""
Activation Maximization Visualization for Ripple Detection CNN

Loads a trained CNN model for ripple detection  and performs
activation maximization to visualize what kind of input pattern causes strong activation
for individual convolutional filters in the network.

Use gradient ascent to synthesize inputs that maximize activation of a specific filter
Visualize these inputs as 8-channel LFP signals
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42) #for reproducibility
dataset = RippleDataset(data_path='data/', annotations_path='annotations/')
model = RipNetCNN()
model_dict =torch.load('best_model_pytorch/best_model.pth', weights_only=True)
model.load_state_dict(model_dict)
model.eval()
# Extract first conv layer (adjust name as needed)
def maximize_activation(model, layer_idx, filter_idx, input_shape, steps=200, lr=1e-3):
    # Start from random input
    input_tensor = torch.randn(input_shape, requires_grad=True, device='cpu')  # or 'cuda'

    # Hook to capture activation from the chosen filter
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output

    hook = model.features[layer_idx].register_forward_hook(hook_fn)

    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        model(input_tensor)
        # Extract activation from selected filter
        act = activation[0, filter_idx]  # shape: [H, W]
        loss = -act.mean()  # Negative so we maximize
        loss.backward()
        optimizer.step()

    hook.remove()

    return input_tensor.detach(), act.detach()

def plot_8_channel_traces(input_tensor, title=None):
    channels = input_tensor.shape[2]
    timepoints = input_tensor.shape[3]

    fig, axs = plt.subplots(channels, 1, figsize=(12, 2*channels), sharex=True)
    for ch in range(channels):
        axs[ch].plot(input_tensor[0, 0, ch].numpy())
        axs[ch].set_ylabel(f'Ch {ch}')
        axs[ch].grid(True)
    axs[-1].set_xlabel('Timepoints')
    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Example usage:
input_shape = (1, 1, 8, 416)  # batch, channel, 8 LFP channels, time

for layer_idx in [0,3]:  # conv layer 3 and 4 as per your model.features
    n_filters = model.features[layer_idx].out_channels
    print(f"Layer {layer_idx} has {n_filters} filters")

    # Plot first 3 filters for each layer to keep it manageable
    for filter_idx in range(min(3, n_filters)):
        max_input, activation = maximize_activation(model, layer_idx, filter_idx, input_shape)
        plot_8_channel_traces(max_input, title=f"Layer {layer_idx} Filter {filter_idx} Maximally Activating Input")