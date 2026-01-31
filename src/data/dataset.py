import torch
from torch.utils.data import Dataset
from pathlib import Path


class MAESTRODataset(Dataset):
    def __init__(self, cache_dir, split, sequence_length=None):
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / split
        self.sequence_length = sequence_length
        self.files = sorted(list(self.split_dir.glob("*.pt")))
        
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {self.split_dir}")
        
        # Load normalization stats
        stats = torch.load(self.cache_dir / "stats.pt")
        self.mean = stats["mean"]
        self.std = stats["std"]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        spec = data["spec"]
        frame_labels = data["frame_labels"]
        onset_labels = data["onset_labels"]
        
        # Normalize
        spec = (spec - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        
        num_frames = spec.shape[-1]
        
        if self.sequence_length is not None and num_frames > self.sequence_length:
            start = torch.randint(0, num_frames - self.sequence_length, (1,)).item()
            end = start + self.sequence_length
            
            spec = spec[:, start:end]
            frame_labels = frame_labels[start:end]
            onset_labels = onset_labels[start:end]
        
        return {
            "spec": spec,
            "frame_labels": frame_labels,
            "onset_labels": onset_labels,
        }