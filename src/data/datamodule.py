import lightning as L
from torch.utils.data import DataLoader

from src.data.dataset import MAESTRODataset

class MAESTRODataModule(L.LightningDataModule):
    def __init__(self, cache_dir, sequence_length, batch_size, num_workers=4): #TODO double check workers
        super().__init__()
        self.cache_dir = cache_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MAESTRODataset(
                self.cache_dir, 
                split="train", 
                sequence_length=self.sequence_length
            )
            self.val_dataset = MAESTRODataset(
                self.cache_dir, 
                split="validation", 
                sequence_length=self.sequence_length # fixed length for batching
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = MAESTRODataset(
                self.cache_dir, 
                split="test", 
                sequence_length=None # full sequences
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # full sequences, variable length
            shuffle=False,
            num_workers=self.num_workers
        )