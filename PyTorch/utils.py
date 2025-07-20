#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:09:03 2025

@author: claire
lowpass: barbara imbrosci
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pickle
from scipy import signal

def find_events(binary_preds, min_duration=int(0.02 * 1250)):
    """
    Given a 1D binary prediction vector (0s and 1s),
    return a list of (start_idx, stop_idx) tuples for each contiguous event of 1s.
    Use on input labels to get ground truth ripple start and stop times
    """
    preds = binary_preds.flatten()  # ensure 1D
    events = []
    start = None

    for i, val in enumerate(preds):
        if val == 1 and start is None:
            start = i  # start of a new event
        elif val == 0 and start is not None:
            events.append((start, i - 1))  # end of event
            start = None

    # Handle case where event goes till the end
    if start is not None:
        events.append((start, len(preds) - 1))

    return events

def find_continuous_events(preds, threshold=0.5, min_duration=int(0.02 * 1250), smoothing_window=5):
    """
    Detect events in continuous prediction signal.
    Events are regions where predictions exceed `threshold` for at least `min_duration` samples (20 ms).
    Outputs list of (start_idx, stop_idx) tuples.
    """
    #threshold should be float and preds should be np array
    threshold = float(threshold.item()) if torch.is_tensor(threshold) else float(threshold)    # Convert to 1D numpy array
    if isinstance(preds, torch.Tensor):
        preds_np= preds.detach().cpu().numpy().flatten()
    elif isinstance(preds,np.ndarray):
        preds_np=preds.flatten()
    smoothed = np.convolve(preds_np, np.ones(smoothing_window)/smoothing_window, mode='same') #smooth by convolution
    
    binary_preds = (smoothed >= threshold).astype(int) #get binary trace indicating above and below threshold
    events = []
    start = None
    #dynamically track length of segments above threshold, save start and stops if right event length
    for i, val in enumerate(binary_preds):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i - 1))
            start = None
    if start is not None and len(preds) - start >= min_duration:
        events.append((start, len(preds) - 1))
    
    return events

def ripple_accuracy(y_trues, y_preds, find_events, find_peak_times,threshold=0.5, tolerance_ratio=0.5):
    """
    Takes model predictions and labels and returns model performance (precision,recall,f1)
    Assumes predictions (y_preds) and labels (y_trues) are still tensors on gpu
    Checks whether predictions start/stop times are close enough (within tolerance_ratio) of true event start/stop times
    """
    total_TP = total_FP = total_FN = 0

    for y_true, y_pred in zip(y_trues, y_preds): #zip combines 2 iterables pair-wise
        y_true=y_true.detach().cpu().numpy().flatten()
        y_pred=y_pred.detach().cpu().numpy().flatten()
        
        #get start stop times of predicted and GT ripples
        true_events = find_events(y_true) 
        pred_events = find_continuous_events(y_pred,threshold=threshold) 

        #keep track of which true and predicted events have already been matched
        matched_true = set()
        matched_pred = set()
        
        for i, (t_start, t_stop) in enumerate(true_events):
            duration = t_stop - t_start
            tolerance = int(duration * tolerance_ratio)
            # for each true event check if corresponding pred event within tolerance limits
            for j, (p_start, p_stop) in enumerate(pred_events):
                if j in matched_pred: # already got it
                    continue
                if (
                    abs(p_start - t_start) <= tolerance and
                    abs(p_stop - t_stop) <= tolerance
                ):
                    matched_true.add(i) #add to set
                    matched_pred.add(j) #add to set
                    break
            
            TP = len(matched_true)
            FP = len(pred_events) - TP
            FN = len(true_events) - TP

        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'true_positives': total_TP,
        'false_positives': total_FP,
        'false_negatives': total_FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_predictions_vs_truth(preds, labels, threshold=0.5, smoothing_window=5, min_duration=0.02 * 1250, idx=None):
    """
    Visualize model predictions vs ground truth events.

    Args:
        preds: 1D tensor or numpy array of predicted probabilities.
        labels: 1D tensor or numpy array of ground truth binary labels.
        threshold: threshold for detecting predicted events.
        smoothing_window: window size for smoothing predictions.
        min_duration: minimum length of event in samples.
        idx: optional index for labeling figure.
    """
    
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    preds_np = preds_np.flatten()
    labels_np = labels_np.flatten()

    # Smoothing
    smoothed = np.convolve(preds_np, np.ones(smoothing_window) / smoothing_window, mode='same')
    binary_preds = (smoothed >= threshold).astype(int)

    # Find events
    def find_events(binary_vec):
        events = []
        start = None
        for i, v in enumerate(binary_vec):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                if i - start >= min_duration:
                    events.append((start, i))
                start = None
        if start is not None and len(binary_vec) - start >= min_duration:
            events.append((start, len(binary_vec)))
        return events

    true_events = find_events(labels_np)
    pred_events = find_events(binary_preds)

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(preds_np, label='Raw Predictions', alpha=0.5)
    plt.plot(smoothed, label='Smoothed Predictions', color='blue')
    plt.plot(labels_np, label='Ground Truth', color='green', alpha=0.5)
    plt.axhline(threshold, color='gray', linestyle='--', label=f'Threshold={threshold}')

    for start, end in true_events:
        plt.axvspan(start, end, color='green', alpha=0.2, label='True Event')
    for start, end in pred_events:
        plt.axvspan(start, end, color='red', alpha=0.2, label='Predicted Event')

    plt.title(f'Predictions vs Ground Truth{" (Example "+str(idx)+")" if idx is not None else ""}')
    plt.xlabel('Time')
    plt.ylabel('Probability / Label')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    plt.show()

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics_over_thresholds(thresholds, all_labels, all_preds,
                                     find_events, find_peak_times,
                                     tolerance_ratio=0.3):
    """
    Evaluate model performance across a range of thresholds. Used epoch-wise in validation phase of training.

    For each threshold, compute F1 score, precision, and recall by comparing predicted
    ripple events (derived from model outputs) with ground truth labels. 
    Identify the threshold that yields the highest F1 score. Output best f1 and threshold by epoch during training. 

    Returns:
        metric_data (dict): Dictionary with lists of F1 scores, precision, and recall values per threshold.
        best_thresh (float): Threshold with the highest F1 score.
        best_f1 (float): Highest F1 score achieved across thresholds.
    """
    best_f1 = 0
    best_thresh = 0.5 #Store threshold that gives highest F1

    metric_data = {
        'thresholds': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': []
    }
    
    #for each threshold, get performance metrics, store in dictionary metric_data
    for thresh in thresholds:
        metrics = ripple_accuracy(
            all_labels, all_preds,
            find_events, find_peak_times,
            threshold=thresh,
            tolerance_ratio=tolerance_ratio
        )

        f1 = metrics['f1_score']
        precision = metrics['precision']
        recall = metrics['recall']

        metric_data['thresholds'].append(thresh)
        metric_data['f1_scores'].append(f1)
        metric_data['precisions'].append(precision)
        metric_data['recalls'].append(recall)
        #if higher f1 is found than best_f1, update it and update corresponding thresh as best_thresh
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return metric_data, best_thresh, best_f1

def plot_metrics_over_thresholds(metrics_per_epoch, metric_names=['f1_scores', 'precisions', 'recalls']):
    """
    Plots metric curves (F1, precision, recall) vs threshold for each epoch.

    metrics_per_epoch (list of dict): List where each dict contains metrics collected
                                          across different thresholds for a single epoch.
                                          Each dict should contain:
                                              - 'thresholds': list of threshold values
                                              - 'f1_scores': list of F1 scores at each threshold
                                              - 'precisions': list of precisions at each threshold
                                              - 'recalls': list of recalls at each threshold
                                              - 'epoch': epoch number
        metric_names (list of str): Names of metrics to plot (default: F1, precision, recall).

    Saves one plot per metric as "<metric_name>_vs_threshold.png"
    """

    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Plot the metric for each epoch
        for epoch_data in metrics_per_epoch:
            plt.plot(epoch_data['thresholds'], epoch_data[metric_name],
                     label=f"Epoch {epoch_data['epoch']}")
        
        # Title and axis formatting
        plt.title(f'{metric_name.replace("_", " ").title()} vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel(metric_name.replace('_', ' ').title())
        
        # Legend outside the plot for clarity
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save and show the figure
        plt.savefig(f"{metric_name}_vs_threshold.png")
        plt.show()
    
def plot_example_preds():
    
    """
    Plots example predictions vs ground truth labels for ripple detection.
    Displays both the LFP input (8 channels) and predicted ripple probabilities,
    with event spans for ground truth and predicted events.

    """

    fs = 1250  # Sampling rate in Hz
    offset = 2.5  # Vertical offset for plotting LFP channels
    threshold = 0.7  # Threshold for converting predicted probabilities to events

    # Load model outputs and corresponding labels
    with open("model_output_pkl/batch_output.pkl", "rb") as f:
        data = pickle.load(f)

    inputs = data["inputs"]   # Shape: [N, 1, 8, 416] or [N, 8, 416]
    labels = data["labels"]   # Shape: [N, 416]
    preds = data["preds"]     # Shape: [N, 416]

    # Loop through selected example indices (every 3rd index up to 127)
    for example_idx in np.arange(0, 127, 3):

        lfp = np.squeeze(inputs[example_idx])  # Shape becomes [8, 416]
        label = labels[example_idx]            # Ground truth binary labels
        pred = preds[example_idx]              # Predicted probabilities

        # Find predicted event spans (start/stop) based on threshold
        events_pred = find_continuous_events(
            pred, threshold=threshold,
            min_duration=int(0.02 * fs),
            smoothing_window=5
        )
        events_pred_ms = [(start / fs * 1000, stop / fs * 1000) for (start, stop) in events_pred]

        # Get ground truth event spans
        events_gt = find_events(label)
        events_gt_ms = [(start / fs * 1000, stop / fs * 1000) for (start, stop) in events_gt]

        # Time axis in milliseconds
        time = np.arange(lfp.shape[1]) / fs * 1000

        # Create two subplots: one for LFP, one for predictions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # --- Plot LFP channels ---
        for ch in range(lfp.shape[0]):
            ax1.plot(time, lfp[ch] + ch * offset, color='black')
        ax1.set_ylabel("LFP Channels")
        ax1.set_title("LFP Input (8 channels)")
        ax1.tick_params(labelleft=False)

        # --- Plot predictions and labels ---
        ax2.plot(time, pred, label='Prediction', alpha=0.7)
        ax2.plot(time, label, label='Ground Truth', alpha=0.7)

        # Overlay ground truth event spans
        if events_gt_ms:
            for start, stop in events_gt_ms:
                ax2.axvspan(start, stop, color='green', alpha=0.2, label='GT Event')

        # Overlay predicted event spans
        if events_pred_ms:
            for start, stop in events_pred_ms:
                ax2.axvspan(start, stop, color='red', alpha=0.2, label='Predicted Event')

        # Plot formatting
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_title("Prediction vs Ground Truth")
        ax2.axhline(threshold, linestyle='--', color='gray', label=f"Threshold={threshold}")

        # Combine duplicate legend entries
        handles, lbls = ax2.get_legend_handles_labels()
        unique = dict(zip(lbls, handles))
        ax2.legend(unique.values(), unique.keys())

        plt.tight_layout()
        plt.show()
        plt.close()
   
def lowpass(data, cutoff_frequency, fs=20000, order=2):
    """
    Low pass filters the input signal.

    inputs:
        data = the signal to be filtered. It needs to have shape [-1, 1]
        cutoff_frequency = the cutoff frequency
        fs = the sampling rate (Hz)
        order = polynomial
    output:
        the filtered_data
    """
    nyq = fs * 0.5
    b, a = signal.butter(order, cutoff_frequency / nyq)
    if len(data.shape) > 1:
        if data.shape[0] > data.shape[1]:
            ref_filt = signal.lfilter(b, a, data, axis=0)
        else:
            ref_filt = signal.lfilter(b, a, data, axis=1)
    else:
        ref_filt = signal.lfilter(b, a, data, axis=0)
    return ref_filt

def merge_events(events, merge_gap=0):
    """
    Merge overlapping or near-overlapping events for inference.

        events: List of (start, stop) tuples (in sample indices or ms).
        merge_gap: Maximum allowed gap between events to merge them (e.g., 20 ms or samples).

        returns merged list of (start, stop) tuples.
    """
    if not events:
        return []
    
    # Sort events by start time
    events = sorted(events, key=lambda x: x[0])
    merged = [events[0]]

    for current in events[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end + merge_gap:
            # Overlap or within gap -> merge
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged
    
