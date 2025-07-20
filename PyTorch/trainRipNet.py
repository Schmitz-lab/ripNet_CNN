#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 08:27:12 2025

@author: claire
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import (
    find_events, find_continuous_events, find_peak_times,ripple_accuracy, set_seed,
    compute_metrics_over_thresholds, plot_metrics_over_thresholds
 )

def train_model(model, train_loader, val_loader, test_loader,pos_weight, threshold =0.5, epochs=20, lr=1e-3):
    """
   Train a CNN model for ripple detection with model checkpointing based on best validation F1 score,
   F1-based threshold tuning, and evaluation.
   """
   #training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)  # add weight to rebalance data 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.001)
    best_model_wts = None
    save_dir = "best_model_pytorch/"
    
    #Variables to keep track of
    train_losses =[]
    val_losses=[]
    best_f1 = 0
    metrics_per_epoch = [] 

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 20)

        # --- Training Phase ---
        model.train()  # set model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #reset gradients from last epoch
            outputs = model(inputs) # in logits
            loss = loss_fn(outputs, labels)
            loss.backward() #compute gradients 
            optimizer.step() #update weights
            #loss.item() gives mean loss per batch..multiply by batch size to get total loss for each batch
            running_loss += loss.item() * inputs.size(0) 
        epoch_train_loss = running_loss / len(train_loader.dataset) # important when last batch size is smaller
        train_losses.append(epoch_train_loss)
        print(f'Train Loss: {epoch_train_loss:.4f}')

# --- Validation Phase ---
        model.eval()  # set model to eval mode
        running_val_loss = 0.0
        all_preds=[]
        all_labels=[]
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                outputs_probs = torch.sigmoid(outputs) #convert logits to probabilities
                all_preds.append(outputs_probs)
                all_labels.append(labels)
                running_val_loss += loss.item() * inputs.size(0)
         
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        thresholds = np.arange(0.1, 1, 0.05) #test a range of thresholds
        
        #compute performance across range of thresholds
        metric_data, best_thresh, current_f1 = compute_metrics_over_thresholds(thresholds,
            all_labels, all_preds, find_events, find_peak_times)
        
        # Store for plotting
        metric_data['epoch'] = epoch + 1
        metrics_per_epoch.append(metric_data)
        
        print(f"Best F1 this epoch: {current_f1:.3f} at threshold: {best_thresh:.2f}")
        #if the best f1 of this epoch is better than previous ones, store it and the corresponding threshold  
        #save this version of the model
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_wts = model.state_dict().copy()
            print("Best model updated!")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(best_model_wts, save_path)
            print(f"Best model saved to {save_path}")
            
    # --- Testing Phase ---
    print('\nTesting best model on test data...')
    model.load_state_dict(best_model_wts)
    model.eval() #no dropout e.g.
    running_test_loss = 0.0
    
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=model(inputs)
            loss = loss_fn(outputs, labels)

            outputs_probs = torch.sigmoid(outputs)
            all_preds.append(outputs_probs)
            all_labels.append(labels)
            running_test_loss += loss.item() * inputs.size(0)
    

    test_loss = running_test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    #using threshold associated with best f1, compute metrics on testing set
    metrics=ripple_accuracy(all_labels, all_preds, find_events, find_peak_times,threshold=best_thresh, tolerance_ratio=0.3)    
    print(f'Precision: {metrics["precision"]:.2f}, Recall: {metrics["recall"]:.2f}, F1: {metrics["f1_score"]:.2f}')

    #Plot loss curves
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig('plots/loss_curve.png')
    plt.show() 
    
    
    plot_metrics_over_thresholds(metrics_per_epoch)

    return model, metrics

