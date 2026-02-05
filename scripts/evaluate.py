import hydra
from omegaconf import DictConfig
import torch
import numpy as np

from src.data.datamodule import MAESTRODataModule
from src.lightning_module import AMTLightningModule
from src.evaluation import (
    optimize_threshold, 
    evaluate_frame_metrics,
    evaluate_note_level,
    extract_ref_notes_from_labels
)
from src.postprocess import decode_notes


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
    dm.setup()
    
    # Load model
    model = AMTLightningModule.load_from_checkpoint(cfg.paths.checkpoint_path)
    model.to(device)
    model.eval()
    
    model_type = model.hparams.model_type
    print(f"Evaluating {model_type.upper()} model...")
    
    # frame-level eval first
    print("\n=== Frame-level Evaluation ===")
    threshold, val_f1 = optimize_threshold(model, dm.val_dataloader(), device)
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Validation F1: {val_f1:.3f}")
    
    metrics = evaluate_frame_metrics(model, dm.test_dataloader(), device, threshold)
    print(f"Test Precision: {metrics['precision']:.3f}")
    print(f"Test Recall: {metrics['recall']:.3f}")
    print(f"Test F1: {metrics['f1']:.3f}")
    
    # note-level eval for crnn
    if model_type == "crnn":
        print("\n=== Note-level Evaluation ===")
        
        all_pred_notes = []
        all_ref_notes = []
        
        with torch.no_grad():
            for batch in dm.test_dataloader():
                spec = batch["spec"].to(device)
                frame_labels = batch["frame_labels"].numpy()
                onset_labels = batch["onset_labels"].numpy()
                
                output = model(spec)
                frame_probs = torch.sigmoid(output["frame"]).cpu().numpy()
                onset_probs = torch.sigmoid(output["onset"]).cpu().numpy()
                
                # process each item
                for i in range(spec.shape[0]):
                    pred_notes = decode_notes(
                        frame_probs[i], onset_probs[i],
                        frame_thresh=threshold, onset_thresh=threshold,
                        fps=fps
                    )
                    ref_notes = extract_ref_notes_from_labels(
                        frame_labels[i], onset_labels[i], fps=fps
                    )
                    
                    all_pred_notes.extend(pred_notes)
                    all_ref_notes.extend(ref_notes)
        
        note_metrics = evaluate_note_level(all_pred_notes, all_ref_notes)
        print(f"Onset Precision: {note_metrics['onset_precision']:.3f}")
        print(f"Onset Recall: {note_metrics['onset_recall']:.3f}")
        print(f"Onset F1: {note_metrics['onset_f1']:.3f}")
        print(f"Offset Precision: {note_metrics['offset_precision']:.3f}")
        print(f"Offset Recall: {note_metrics['offset_recall']:.3f}")
        print(f"Offset F1: {note_metrics['offset_f1']:.3f}")


if __name__ == "__main__":
    main()