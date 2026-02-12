import hydra
import torch
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule


def objective(trial, cfg):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    sequence_length = trial.suggest_categorical("sequence_length", [625, 1000])
    
    if cfg.model.name == "crnn":
        pos_weight = trial.suggest_float("pos_weight", 4.0, 7.0)
        onset_weight = trial.suggest_float("onset_weight", 0.5, 2.5)
    else:
        pos_weight = trial.suggest_float("pos_weight", 1.0, 4.0)
        onset_weight = 1.0
    
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    
    model = AMTLightningModule(
        n_bins=cfg.audio.get("n_bins", cfg.audio.get("n_mels")),
        model_type=cfg.model.name,
        num_pitches=cfg.model.num_pitches,
        channels=list(cfg.model.channels),
        pool_sizes=list(cfg.model.pool_sizes),
        lstm_hidden=cfg.model.get("lstm_hidden", 256),
        lstm_layers=cfg.model.get("lstm_layers", 2),
        bidirectional=cfg.model.get("bidirectional", True),
        dropout=dropout,
        projection_dim=cfg.model.get("projection_dim", 512),
        lr=lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=pos_weight,
        onset_weight=onset_weight
    )
    
    # if old CNN is used as frozen backbone
    if cfg.model.get("pretrained_cnn_path"):
        
        ckpt = torch.load(cfg.model.pretrained_cnn_path, map_location="cpu", weights_only=False)
        backbone_weights = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if "conv_blocks" in k
        }
        model.model.load_state_dict(backbone_weights, strict=False)
        if cfg.model.get("freeze_backbone", False):
            for param in model.model.conv_blocks.parameters():
                param.requires_grad = False
    
    callbacks = [
        EarlyStopping(monitor="val_f1", mode="max", patience=10),
        PyTorchLightningPruningCallback(trial, monitor="val_f1")
    ]
    
    trainer = L.Trainer(
        max_epochs=25,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False
    )
    
    trainer.fit(model, dm)
    
    return trainer.callback_metrics["val_f1"].item()


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(lambda trial: objective(trial, cfg), n_trials=20, show_progress_bar=True)
    
    print("\n=== Best Trial ===")
    print(f"Value (F1): {study.best_trial.value:.3f}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()