import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from src.data.transforms import AudioTransform
from src.data.labels import MIDILabels


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    maestro_dir = Path(cfg.paths.maestro_dir)
    cache_dir = Path(cfg.paths.cache_dir) / cfg.audio.name
    
    metadata = pd.read_csv(maestro_dir / "maestro-v3.0.0.csv")
    
    audio_transform = AudioTransform(cfg.audio)
    midi_labels = MIDILabels(cfg.audio)
    
    # First pass: save spectrograms (without normalization)
    for split in ["train", "validation", "test"]:

        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        split_df = metadata[metadata["split"] == split]
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split):

            audio_path = maestro_dir / row["audio_filename"]
            midi_path = maestro_dir / row["midi_filename"]
            
            spec = audio_transform(audio_path)
            num_frames = spec.shape[-1]
            frame_labels, onset_labels = midi_labels(midi_path, num_frames)
            
            save_name = Path(row["audio_filename"]).stem + ".pt"
            torch.save({
                "spec": spec,
                "frame_labels": torch.from_numpy(frame_labels),
                "onset_labels": torch.from_numpy(onset_labels),
            }, split_dir / save_name)
    
    # Second pass: compute stats from training set only
    print("Computing normalization statistics...")
    train_files = list((cache_dir / "train").glob("*.pt"))
    
    n_bins = cfg.audio.n_mels if cfg.audio.name == "mel" else cfg.audio.n_bins
    total_sum = torch.zeros(n_bins)
    total_sq_sum = torch.zeros(n_bins)
    total_frames = 0
    
    for f in tqdm(train_files, desc="stats"):
        spec = torch.load(f)["spec"]
        total_sum += spec.sum(dim=-1)
        total_sq_sum += (spec ** 2).sum(dim=-1)
        total_frames += spec.shape[-1]
    
    mean = total_sum / total_frames
    std = torch.sqrt(total_sq_sum / total_frames - mean ** 2)
    
    torch.save({"mean": mean, "std": std}, cache_dir / "stats.pt")
    print(f"Done! Stats saved to {cache_dir / 'stats.pt'}")


if __name__ == "__main__":
    main()