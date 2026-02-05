import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule


def objective(trial, cfg):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    pos_weight = trial.suggest_float("pos_weight", 5.0, 50.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    if cfg.model.name == "crnn":
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [128, 256, 512])
        onset_weight = trial.suggest_float("onset_weight", 0.5, 2.0)
    else: # cnn
        lstm_hidden = 256
        onset_weight = 1.0
    
    # DataModule
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers
    )
    
    # Model
    model = AMTLightningModule(
        n_bins=cfg.audio.get("n_bins", cfg.audio.get("n_mels")),
        model_type=cfg.model.name,
        num_pitches=cfg.model.num_pitches,
        channels=cfg.model.channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=cfg.model.get("lstm_layers", 2),
        dropout=dropout,
        lr=lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=pos_weight,
        onset_weight=onset_weight
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val/f1", mode="max", patience=5),
        PyTorchLightningPruningCallback(trial, monitor="val/f1")
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        precision="16-mixed",
        callbacks=callbacks,
        enable_progress_bar=False,
        logger=False
    )
    
    trainer.fit(model, dm)
    
    return trainer.callback_metrics["val/f1"].item()


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(lambda trial: objective(trial, cfg), n_trials=20)
    
    print("\n=== Best Trial ===")
    print(f"Value (F1): {study.best_trial.value:.3f}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()