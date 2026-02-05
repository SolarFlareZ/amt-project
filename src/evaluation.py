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






# NOT YET ADDED TO SCRIPT

def evaluate_note_level(pred_notes, ref_notes, onset_tolerance=0.05, offset_ratio=0.2):
    if len(pred_notes) == 0:
        pred_intervals = np.zeros((0, 2))
        pred_pitches = np.zeros(0)
    else:
        pred_intervals = np.array([[n[1], n[2]] for n in pred_notes])
        pred_pitches = np.array([n[0] for n in pred_notes])
    
    if len(ref_notes) == 0:
        ref_intervals = np.zeros((0, 2))
        ref_pitches = np.zeros(0)
    else:
        ref_intervals = np.array([[n[1], n[2]] for n in ref_notes])
        ref_pitches = np.array([n[0] for n in ref_notes])
    
    # midi to hz for mir_eval
    pred_pitches_hz = 440 * (2 ** ((pred_pitches - 69) / 12)) if len(pred_pitches) > 0 else np.zeros(0)
    ref_pitches_hz = 440 * (2 ** ((ref_pitches - 69) / 12)) if len(ref_pitches) > 0 else np.zeros(0)
    
    # onset only
    onset_p, onset_r, onset_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches_hz,
        pred_intervals, pred_pitches_hz,
        onset_tolerance=onset_tolerance,
        offset_ratio=None
    )
    
    # Onset + offset
    offset_p, offset_r, offset_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches_hz,
        pred_intervals, pred_pitches_hz,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio
    )
    
    return {
        "onset_precision": onset_p,
        "onset_recall": onset_r,
        "onset_f1": onset_f,
        "offset_precision": offset_p,
        "offset_recall": offset_r,
        "offset_f1": offset_f,
    }


def extract_ref_notes_from_labels(frame_labels, onset_labels, fps=31.25):
    notes = []
    num_frames, num_pitches = frame_labels.shape
    
    for pitch in range(num_pitches):
        frames = frame_labels[:, pitch] > 0.5
        onsets = onset_labels[:, pitch] > 0.5
        
        onset_frames = np.where(onsets)[0]
        
        for onset_frame in onset_frames:
            offset_frame = onset_frame + 1
            while offset_frame < num_frames and frames[offset_frame]:
                offset_frame += 1
            
            onset_time = onset_frame / fps
            offset_time = offset_frame / fps
            midi_pitch = pitch + 21
            
            notes.append((midi_pitch, onset_time, offset_time))
    
    return notes