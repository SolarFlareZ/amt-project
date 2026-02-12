import hydra
from omegaconf import DictConfig
import torch

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule
from src.evaluation import optimize_threshold, evaluate_frame_metrics, evaluate_note_level_dataset


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    dm.setup()
    
    model = AMTLightningModule.load_from_checkpoint(
        cfg.paths.checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()
    
    model_type = model.hparams.model_type
    print(f"Evaluating {model_type.upper()} model...")
    
    # Frame-level
    print("\n=== Frame-level Evaluation ===")
    use_chunks = cfg.audio.name == "stft"
    threshold, val_f1 = optimize_threshold(model, dm.val_dataloader(), device, use_chunks=use_chunks)
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Validation F1: {val_f1:.3f}")
    
    metrics = evaluate_frame_metrics(model, dm.test_dataloader(), device, threshold, use_chunks=use_chunks)
    print(f"Test Precision: {metrics['precision']:.3f}")
    print(f"Test Recall: {metrics['recall']:.3f}")
    print(f"Test F1: {metrics['f1']:.3f}")
    
    # Note-level for crnn
    if model_type == "crnn":
        print("\n=== Note-level Evaluation ===")
        note_metrics = evaluate_note_level_dataset(model, dm.test_dataloader(), device, threshold, fps, use_chunks=use_chunks)
        print(f"Onset F1: {note_metrics['onset_f1']:.3f}")
        print(f"Offset F1: {note_metrics['offset_f1']:.3f}")


if __name__ == "__main__":
    main()