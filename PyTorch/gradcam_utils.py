#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:51:30 2025

@author: claire
"""

import torch
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, device, target_layer):
    """
    Generate Grad-CAM for a specific layer.
    Get activations (for that layer) and gradients from the timepoint of highest ripple probability (in target layer)
    Get mean gradient per channel (scalar) and use it to weight activations
    Sum across channels, ReLU to only get positive contributions
    Upsample to fit onto original input for plotting

    Returns:
        cam: numpy array of shape [8, 416], Grad-CAM heatmap.
    """
    activations = {}
    gradients = {}
    #when forward hook is registered, store activations
    def forward_hook(module, input, output): 
        activations["value"] = output.detach()
        #when backward hook is registered, store gradients
    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.zero_grad()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    output = model(input_tensor)  # [1, 416]
    probs = torch.sigmoid(output)

    target_time_idx = probs[0].argmax().item() #get timepoint with highest ripple prob.

    #Backward pass: gradients of score w.r.t activations
    score = output[0, target_time_idx] #in logits
    score.backward()

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()

    fmap = activations["value"]       # shape: [1, C, H, W]
    grads = gradients["value"]        # same shape
    
    
    pooled_grads = torch.mean(grads, dim=(2, 3))  # [1, C] #each channel gets a scalar weight
    weighted_fmap = fmap[0] * pooled_grads[0].unsqueeze(1).unsqueeze(2)  # [C, H, W] weight each feature map
    cam = weighted_fmap.sum(dim=0)   # [H, W] sum across channels

    cam = torch.clamp(cam, min=0) #reLU to get only positive signals

    # Resize to input shape
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(8, 416), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam