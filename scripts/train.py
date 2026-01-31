import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    model = AMTLightningModule(
        n_mels=cfg.audio.n_mels,
        num_pitches=cfg.model.num_pitches,
        channels=cfg.model.channels,
        dropout=cfg.model.dropout,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=cfg.train.pos_weight
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename="best-{epoch}-{val/f1:.3f}",
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

    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=dict(cfg)
    )

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