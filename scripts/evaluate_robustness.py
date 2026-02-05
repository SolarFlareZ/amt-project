import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule
from src.evaluation import evaluate_frame_metrics, evaluate_note_level, extract_ref_notes_from_labels
from src.postprocess import decode_notes
from src.data.corruptions import AddNoise, AddReverb, Detune


def evaluate_with_corruption(model, dataloader, device, corruption, fps, threshold=0.5):
    model.eval()
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            spec = batch["spec"]
            
            # Apply corruption if provided
            if corruption is not None:
                spec = torch.stack([corruption(s) for s in spec])
            
            spec = spec.to(device)
            labels = batch["frame_labels"].to(device)
            
            output = model(spec)
            if isinstance(output, dict):
                logits = output["frame"]
            else:
                logits = output
            
            preds = (torch.sigmoid(logits) > threshold).float()
            
            total_tp += ((preds == 1) & (labels == 1)).sum().item()
            total_fp += ((preds == 1) & (labels == 0)).sum().item()
            total_fn += ((preds == 0) & (labels == 1)).sum().item()
    
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {"precision": precision, "recall": recall, "f1": f1}


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    
    # DataModule
    dm = MAESTRODataModule(
        cache_dir=f"{cfg.paths.cache_dir}/{cfg.audio.name}",
        sequence_length=cfg.train.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )
    dm.setup("test")
    
    # Load model
    model = AMTLightningModule.load_from_checkpoint(cfg.paths.checkpoint_path)
    model.to(device)
    model.eval()
    
    # Define corruptions
    corruptions = {
        "clean": None,
        "noise_20db": AddNoise(snr_db=20),
        "noise_10db": AddNoise(snr_db=10),
        "reverb_small": AddReverb(decay=0.3, delay_ms=30),
        "reverb_large": AddReverb(decay=0.5, delay_ms=50),
        "detune_+25": Detune(cents=25),
        "detune_-25": Detune(cents=-25),
    }
    
    # Evaluate each
    print(f"\n{'Corruption':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    
    results = {}
    for name, corruption in corruptions.items():
        metrics = evaluate_with_corruption(
            model, dm.test_dataloader(), device, corruption, fps
        )
        results[name] = metrics
        print(f"{name:<20} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f}")
    
    # Summary
    print("\n=== Degradation from Clean ===")
    clean_f1 = results["clean"]["f1"]
    for name, metrics in results.items():
        if name != "clean":
            degradation = (clean_f1 - metrics["f1"]) / clean_f1 * 100
            print(f"{name:<20} {degradation:>+.1f}%")


if __name__ == "__main__":
    main()