#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:35:31 2025

@author: claire
"""
import torch
from torch.utils.data import DataLoader, random_split
from RipNet import RipNetCNN
from utils import set_seed
from ripDataset import RippleDataset
import pickle

"""
Save a batch of inputs, labels and outputs as a pickled dict for future reference.
Use samples from the test set to ensure that interpretability methods (e.g., Grad-CAM) focus on features the model has genuinely learned to generalize.
"""

#Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42) #for reproducibility
dataset = RippleDataset(data_path='data/', annotations_path='annotations/')
model = RipNetCNN()
model_dict = torch.load('best_model_pytorch/best_model.pth', map_location=device, weights_only=True)
model.load_state_dict(model_dict)
model.to(device)

#split dataset into training (70%), validation (15%)  and testing (15%) datasets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

#ensure split occurs in same way as during training
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)#get same batch every time , bigger batch for more examples

#Get model preds
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # No gradient tracking during inference
    inputs, labels = next(iter(test_loader))
    inputs=inputs.to(device)
    outputs = model(inputs)        
    preds = torch.sigmoid(outputs)  
    
inputs_np = inputs.cpu().numpy()
labels_np = labels.numpy()
preds_np = preds.cpu().numpy()

# Save as a dictionary
data_to_save = {
    'inputs': inputs_np,
    'labels': labels_np,
    'preds': preds_np
}

# Save to a file
with open('batch_output.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
