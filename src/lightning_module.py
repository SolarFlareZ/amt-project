import torch
import torch.nn as nn
import lightning as L

from src.models.cnn import PianoCNN

class AMTLightningModule(L.LightningModule):
    def __init__(self, n_mels, num_pitches=88, channels=[32, 64, 128, 256], 
                 dropout=0.3, lr=3e-4, weight_decay=1e-4, pos_weight=10.0):
        super().__init__()
        self.save_hyperparameters()

        self.model = PianoCNN(
            n_mels=n_mels,
            num_pitches=num_pitches,
            channels=channels,
            dropout=dropout
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        spec = batch["spec"]
        frame_labels = batch["frame_labels"]
        
        logits = self(spec)  # (batch, time, 88)
        loss = self.loss_fn(logits, frame_labels)
        
        # Metrics
        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            tp = ((preds == 1) & (frame_labels == 1)).sum()
            fp = ((preds == 1) & (frame_labels == 0)).sum()
            fn = ((preds == 0) & (frame_labels == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return loss, precision, recall, f1
    
    def training_step(self, batch, batch_idx):
        loss, precision, recall, f1 = self._shared_step(batch)
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, precision, recall, f1 = self._shared_step(batch)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/precision", precision)
        self.log("val/recall", recall)
        self.log("val/f1", f1, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1"
            }
        }