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
import os


def train_model(model, train_loader, val_loader, test_loader,pos_weight, epochs=20, lr=1e-3):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)  # or your chosen loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.0001)
    best_model_wts = None

    train_losses =[]
    val_losses=[]
    best_f1 = 0
    save_dir = "best_model_pytorch/"


    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 20)

        # --- Training Phase ---
        model.train()  # set model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
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
                outputs = torch.sigmoid(model(inputs))
                #print(outputs.min(),outputs.max(),outputs.mean())
                all_preds.append(outputs)
                all_labels.append(labels)
                running_val_loss += loss.item() * inputs.size(0)

        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        metrics=ripple_accuracy(all_labels, all_preds, find_events, find_peak_times, tolerance_ratio=0.3)    
        print(f"Metrics: {metrics['f1_score']:.3f}")
        
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_wts = model.state_dict().copy()
            print("Best model updated!")
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(best_model_wts, save_path)
            print(f"Best model saved to {save_path}")
    # --- Testing Phase ---
    print('\nTesting best model on test data...')
    model.load_state_dict(best_model_wts)
    model.eval()
    running_test_loss = 0.0
    
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            all_preds.append(outputs)
            all_labels.append(labels)
            loss = loss_fn(outputs, labels)
            running_test_loss += loss.item() * inputs.size(0)
    

    test_loss = running_test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    metrics=ripple_accuracy(all_labels, all_preds, find_events, find_peak_times, tolerance_ratio=0.3)    
    print(f'Metrics: {metrics}')

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig('loss_curve.png')
    plt.show() 
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))  # sigmoid because you used BCEWithLogitsLoss
    
            # Choose one example from the batch
            sample_idx = 5
            plot_predictions_vs_truth(
                preds=outputs[sample_idx],
                labels=labels[sample_idx],
                threshold=0.4,
                smoothing_window=7,
                idx=i
            )
            if i == 2:  # Just plot 3 examples
                break
    
    
    
    return model    

