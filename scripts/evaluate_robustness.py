import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MAESTRODataset
from src.lightning_module import AMTLightningModule
from src.evaluation import optimize_threshold, evaluate_frame_metrics, evaluate_note_level_dataset
from src.data.corruptions import AddNoise, GainVariation


def apply_corruption(dataloader, corruption):
    for batch in dataloader:
        if corruption is not None:
            batch["spec"] = torch.stack([corruption(s) for s in batch["spec"]])
        yield batch


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    use_chunks = cfg.audio.name == "stft"
    cache_dir = f"{cfg.paths.cache_dir}/{cfg.audio.name}"
    
    model = AMTLightningModule.load_from_checkpoint(
        cfg.paths.checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.to(device)
    model.eval()
    
    model_type = model.hparams.model_type
    
    # cropped test set for speed, should still onverge orrectly due to law of large numbers
    test_dataset = MAESTRODataset(
        cache_dir,
        split="test",
        sequence_length=cfg.train.sequence_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    
    # val for threshold
    val_dataset = MAESTRODataset(
        cache_dir,
        split="validation",
        sequence_length=cfg.train.sequence_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    
    threshold, _ = optimize_threshold(model, val_loader, device, use_chunks=use_chunks)
    print(f"Optimal threshold: {threshold:.2f}")
    
    corruptions = {
        "clean": None,
        "noise_20db": AddNoise(snr_db=20),
        "noise_10db": AddNoise(snr_db=10),
        "noise_5db": AddNoise(snr_db=5),
        "gain_-6db": GainVariation(gain_db=-6),
        "gain_-12db": GainVariation(gain_db=-12),
    }
    
    print(f"\n{'Corruption':<20} {'Precision':>10} {'Recall':>10} {'Frame F1':>10}", end="")
    if model_type == "crnn":
        print(f" {'Onset F1':>10} {'Offset F1':>10}")
    else:
        print()
    print("-" * (52 if model_type != "crnn" else 74))
    
    results = {}
    for name, corruption in corruptions.items():
        print(f"Evaluating: {name}...")
        
        corrupted_loader = apply_corruption(test_loader, corruption)
        metrics = evaluate_frame_metrics(model, corrupted_loader, device, threshold, use_chunks=use_chunks)
        results[name] = metrics
        
        print(f"{name:<20} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f}", end="")
        
        if model_type == "crnn":
            corrupted_loader = apply_corruption(test_loader, corruption)
            note_metrics = evaluate_note_level_dataset(model, corrupted_loader, device, threshold, fps, use_chunks=use_chunks)
            results[name]["onset_f1"] = note_metrics["onset_f1"]
            results[name]["offset_f1"] = note_metrics["offset_f1"]
            print(f" {note_metrics['onset_f1']:>10.3f} {note_metrics['offset_f1']:>10.3f}")
        else:
            print()
    
    print("\n=== Degradation from Clean ===")
    clean_f1 = results["clean"]["f1"]
    for name, metrics in results.items():
        if name != "clean":
            degradation = (clean_f1 - metrics["f1"]) / clean_f1 * 100
            print(f"{name:<20} Frame F1: {degradation:>+.1f}%", end="")
            if model_type == "crnn" and "onset_f1" in metrics:
                onset_deg = (results["clean"]["onset_f1"] - metrics["onset_f1"]) / results["clean"]["onset_f1"] * 100
                print(f"  Onset F1: {onset_deg:>+.1f}%", end="")
            print()


if __name__ == "__main__":
    main()