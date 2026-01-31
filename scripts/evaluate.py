import hydra
from omegaconf import DictConfig
import torch

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule
from src.evaluation import optimize_threshold, evaluate_frame_metrics


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataModule
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    dm.setup()
    
    # Load model
    model = AMTLightningModule.load_from_checkpoint(cfg.paths.checkpoint_path)
    model.to(device)
    
    # Optimize threshold on validation
    print("Finding optimal threshold...")
    threshold, val_f1 = optimize_threshold(model, dm.val_dataloader(), device)
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Validation F1: {val_f1:.3f}")
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    metrics = evaluate_frame_metrics(model, dm.test_dataloader(), device, threshold)
    print(f"Test Precision: {metrics['precision']:.3f}")
    print(f"Test Recall: {metrics['recall']:.3f}")
    print(f"Test F1: {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()