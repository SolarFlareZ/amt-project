import torch
import torch.nn as nn
import lightning as L

from src.models.cnn import PianoCNN
from src.models.crnn import PianoCRNN


class AMTLightningModule(L.LightningModule):
    def __init__(self, n_bins, model_type="cnn", num_pitches=88, channels=[32, 64, 128, 256],
                pool_sizes=None, lstm_hidden=256, lstm_layers=2, dropout=0.3, bidirectional=True,
                use_residual=False, projection_dim=512, lr=3e-4, weight_decay=1e-4, 
                pos_weight=10.0, onset_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == "cnn":
            self.model = PianoCNN(
                n_bins=n_bins,
                num_pitches=num_pitches,
                channels=channels,
                pool_sizes=pool_sizes,
                dropout=dropout,
                use_residual=use_residual
            )
        elif model_type == "crnn":
            self.model = PianoCRNN(
                n_bins=n_bins,
                num_pitches=num_pitches,
                channels=channels,
                pool_sizes=pool_sizes,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                use_residual=use_residual,
                projection_dim=projection_dim
            )
        else:
            raise ValueError(f"invalid model type: {model_type}")
        
        self.model_type = model_type
        self.onset_weight = onset_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        spec = batch["spec"]
        frame_labels = batch["frame_labels"]
        onset_labels = batch["onset_labels"]
        
        output = self(spec)
        
        if self.model_type == "cnn":
            frame_logits = output
            frame_loss = self.loss_fn(frame_logits, frame_labels)
            loss = frame_loss
            onset_loss = torch.tensor(0.0)
        else:
            frame_logits = output["frame"]
            onset_logits = output["onset"]
            frame_loss = self.loss_fn(frame_logits, frame_labels)
            onset_loss = self.loss_fn(onset_logits, onset_labels)
            loss = frame_loss + self.onset_weight * onset_loss

        # frame level
        with torch.no_grad():
            preds = torch.sigmoid(frame_logits) > 0.5
            tp = ((preds == 1) & (frame_labels == 1)).sum()
            fp = ((preds == 1) & (frame_labels == 0)).sum()
            fn = ((preds == 0) & (frame_labels == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "loss": loss,
            "frame_loss": frame_loss,
            "onset_loss": onset_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        
        self.log("train_loss", metrics["loss"], prog_bar=True)
        self.log("train_frame_loss", metrics["frame_loss"])
        self.log("train_onset_loss", metrics["onset_loss"])
        self.log("train_f1", metrics["f1"], prog_bar=True)
        
        return metrics["loss"]
    
    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        
        self.log("val_loss", metrics["loss"], prog_bar=True)
        self.log("val_frame_loss", metrics["frame_loss"])
        self.log("val_onset_loss", metrics["onset_loss"])
        self.log("val_precision", metrics["precision"])
        self.log("val_recall", metrics["recall"])
        self.log("val_f1", metrics["f1"], prog_bar=True)
    
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
                "monitor": "val_f1"
            }
        }