import hydra
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
        n_bins=cfg.audio.n_mels if cfg.audio.name == "mel" else cfg.audio.n_bins,
        model_type=cfg.model.name,
        num_pitches=cfg.model.num_pitches,
        channels=cfg.model.channels,
        lstm_hidden=cfg.model.get("lstm_hidden", 256),
        lstm_layers=cfg.model.get("lstm_layers", 2),
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.get("bidirectional", True),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=cfg.train.pos_weight,
        onset_weight=cfg.train.onset_weight
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename=f"{cfg.model.name}-{{epoch}}-{{val/f1:.3f}}",
            monitor="val/f1",
            mode="max",
            save_top_k=1
        ),
        EarlyStopping(
            monitor="val/f1",
            mode="max",
            patience=cfg.train.patience
        ),
        LearningRateMonitor()
    ]
    
    # Logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.model.name}_{cfg.audio.name}",
        config=dict(cfg)
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger
    )
    
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()