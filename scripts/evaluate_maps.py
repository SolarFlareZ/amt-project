import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MAESTRODataset
from src.lightning_module import AMTLightningModule
from src.evaluation import optimize_threshold, evaluate_frame_metrics, evaluate_note_level_dataset


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    cache_dir = f"{cfg.paths.cache_dir}/{cfg.audio.name}"
    use_chunks = cfg.audio.name == "stft"
    
    model = AMTLightningModule.load_from_checkpoint(
        cfg.paths.checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()
    
    model_type = model.hparams.model_type
    
    # calculate the threshold from maestro still
    val_dataset = MAESTRODataset(cache_dir, split="validation", sequence_length=cfg.train.sequence_length)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    threshold, _ = optimize_threshold(model, val_loader, device, use_chunks=use_chunks)
    print(f"Threshold (from MAESTRO val): {threshold:.2f}")
    
    # evaluation on maps in {dir}/{representation}/maps
    maps_dataset = MAESTRODataset(cache_dir, split="maps", sequence_length=cfg.train.sequence_length)
    maps_loader = DataLoader(maps_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    
    print("\n=== MAPS Frame-level ===")
    metrics = evaluate_frame_metrics(model, maps_loader, device, threshold, use_chunks=use_chunks)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    
    if model_type == "crnn":
        print("\n=== MAPS Note-level ===")
        note_metrics = evaluate_note_level_dataset(model, maps_loader, device, threshold, fps, use_chunks=use_chunks)
        print(f"Onset F1: {note_metrics['onset_f1']:.3f}")
        print(f"Offset F1: {note_metrics['offset_f1']:.3f}")
    
    # caluclate the difference from maestro
    maestro_dataset = MAESTRODataset(cache_dir, split="test", sequence_length=cfg.train.sequence_length)
    maestro_loader = DataLoader(maestro_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    
    print("\n=== MAESTRO Frame-level ===")
    maestro_metrics = evaluate_frame_metrics(model, maestro_loader, device, threshold, use_chunks=use_chunks)
    print(f"Frame F1: {maestro_metrics['f1']:.3f}")
    
    gap = (maestro_metrics['f1'] - metrics['f1']) / maestro_metrics['f1'] * 100
    print(f"\n=== Cross-dataset Gap ===")
    print(f"Frame F1 drop: {gap:.1f}%")


if __name__ == "__main__":
    main()