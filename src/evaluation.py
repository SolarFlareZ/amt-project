import torch
import numpy as np


def optimize_threshold(model, dataloader, device, thresholds=np.arange(0.1, 0.9, 0.05)):
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            spec = batch["spec"].to(device)
            labels = batch["frame_labels"]
            
            logits = model(spec)
            preds = torch.sigmoid(logits).cpu()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        binary_preds = (all_preds > thresh).float()
        
        tp = ((binary_preds == 1) & (all_labels == 1)).sum()
        fp = ((binary_preds == 1) & (all_labels == 0)).sum()
        fn = ((binary_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def evaluate_frame_metrics(model, dataloader, device, threshold=0.5):
    model.eval()
    model.to(device)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            spec = batch["spec"].to(device)
            labels = batch["frame_labels"].to(device)
            
            logits = model(spec)
            preds = (torch.sigmoid(logits) > threshold).float()
            
            total_tp += ((preds == 1) & (labels == 1)).sum().item()
            total_fp += ((preds == 1) & (labels == 0)).sum().item()
            total_fn += ((preds == 0) & (labels == 1)).sum().item()
    
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"precision": precision, "recall": recall, "f1": f1}