#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:09:03 2025

@author: claire
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def find_events(binary_preds, min_duration=int(0.02 * 1250)):
    """
    Given a 1D binary prediction vector, return list of (start_idx, stop_idx) for each event
    that is at least min_duration long.
    """
   
    preds = binary_preds  # assume numpy array or similar

    preds = preds.flatten()
    events = []
    start = None
    for i, val in enumerate(preds):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start >= min_duration:
                events.append((start, i - 1))
            start = None
    # Check for event reaching to the end
    if start is not None and len(preds) - start >= min_duration:
        events.append((start, len(preds) - 1))
    return events

def find_continuous_events(preds, threshold=0.5, min_duration=int(0.02 * 1250), smoothing_window=5):
    """
    Detect events in continuous prediction signal.
    Events are regions where predictions exceed `threshold` for at least `min_duration` samples.
    
    Args:
        preds (1D numpy array or torch tensor): Continuous prediction scores.
        threshold (float): Value above which a region is considered an event.
        min_duration (int): Minimum length of event (in samples).
    
    Returns:
        List of (start_idx, stop_idx) tuples.
    """

    # Convert to 1D numpy array
    if isinstance(preds, torch.Tensor):
        preds_np= preds.detach().cpu().numpy().flatten()
    elif isinstance(preds,np.ndarray):
        preds_np=preds.flatten()
    smoothed = np.convolve(preds_np, np.ones(smoothing_window)/smoothing_window, mode='same')
    
    binary_preds = (smoothed >= threshold).astype(int)
    events = []
    start = None
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

def find_peak_times(events, preds):
    """
    events: list of (start, stop) tuples
    preds: 1D numpy or torch tensor with predicted probabilities
    """
   
    peak_times = []
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds

    for start, stop in events:
        peak_idx = start + np.argmax(preds_np[start:stop+1])
        peak_times.append(peak_idx)
    return peak_times

def ripple_accuracy(y_trues, y_preds, find_events, find_peak_times, tolerance_ratio=0.5):
    

    total_TP = total_FP = total_FN = 0

    for y_true, y_pred in zip(y_trues, y_preds):
        y_true = y_true.detach().cpu().numpy().flatten()
        y_pred = y_pred.detach().cpu().numpy().flatten()

        true_events = find_events(y_true)
        pred_events = find_continuous_events(y_pred)

        true_peaks = find_peak_times(true_events, y_true)
        pred_peaks = find_peak_times(pred_events, y_pred)

        matched_true = set()
        matched_pred = set()

        for i, (start, stop) in enumerate(true_events):
            true_peak = (start + stop) // 2
            duration = stop - start
            tolerance = int(duration * tolerance_ratio)
            lower = true_peak - tolerance
            upper = true_peak + tolerance

            for j, pred_peak in enumerate(pred_peaks):
                if j in matched_pred:
                    continue
                if lower <= pred_peak <= upper:
                    matched_true.add(i)
                    matched_pred.add(j)
                    break

        TP = len(matched_true)
        FP = len(pred_peaks) - TP
        FN = len(true_peaks) - TP

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
