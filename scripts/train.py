import hydra
import torch
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # DataModule
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    
    # Model
    model = AMTLightningModule(
        n_bins=cfg.audio.get("n_bins", cfg.audio.get("n_mels")),
        model_type=cfg.model.name,
        num_pitches=cfg.model.num_pitches,
        channels=list(cfg.model.channels),
        pool_sizes=list(cfg.model.pool_sizes),
        dropout=cfg.model.dropout,
        use_residual=cfg.model.get("use_residual", False),
        lstm_hidden=cfg.model.get("lstm_hidden", 256),
        lstm_layers=cfg.model.get("lstm_layers", 2),
        bidirectional=cfg.model.get("bidirectional", True),
        projection_dim=cfg.model.get("projection_dim", 512),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=cfg.train.pos_weight,
        onset_weight=cfg.train.onset_weight
    )

    if cfg.model.get("pretrained_cnn_path"): # use old model as frozen backbone
        print(f"Loading CNN backbone from {cfg.model.pretrained_cnn_path}")
        ckpt = torch.load(cfg.model.pretrained_cnn_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        
        backbone_weights = {}
        for k, v in state_dict.items():
            if "conv_blocks" in k:
                new_key = k.replace("model.", "")
                backbone_weights[new_key] = v
    
        model.model.load_state_dict(backbone_weights, strict=False)
        
        # freeze the loaded backbone
        if cfg.model.get("freeze_backbone", False):
            for name, param in model.model.conv_blocks.named_parameters():
                param.requires_grad = False
            print("CNN backbone frozen")


    
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename="{epoch}-{val_f1:.3f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=cfg.train.patience
        ),
        LearningRateMonitor()
    ]
    
    # Logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.get("name") or f"{cfg.model.name}_{cfg.audio.name}",
        config=dict(cfg)
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger
    )
    
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()