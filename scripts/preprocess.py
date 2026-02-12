import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import pretty_midi

from src.data.transforms import AudioTransform
from src.data.labels import MIDILabels


def preprocess_maestro(cfg, audio_transform, midi_labels, cache_dir):
    dataset_dir = Path(cfg.paths.dataset_dir)
    metadata = pd.read_csv(dataset_dir / "maestro-v3.0.0.csv")
    
    for split in ["train", "validation", "test"]:
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        split_df = metadata[metadata["split"] == split]
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split):
            audio_path = dataset_dir / row["audio_filename"]
            midi_path = dataset_dir / row["midi_filename"]
            
            spec = audio_transform(audio_path)
            num_frames = spec.shape[-1]
            frame_labels, onset_labels = midi_labels(midi_path, num_frames)
            
            save_name = Path(row["audio_filename"]).stem + ".pt"
            torch.save({
                "spec": spec,
                "frame_labels": torch.from_numpy(frame_labels),
                "onset_labels": torch.from_numpy(onset_labels),
            }, split_dir / save_name)


def preprocess_maps(cfg, audio_transform, midi_labels, cache_dir):
    pretty_midi.pretty_midi.MAX_TICK = 1e10 # MAPS midi files are weird/buggy. fix from github issues
    dataset_dir = Path(cfg.paths.dataset_dir)
    maps_cache = cache_dir / "maps"
    maps_cache.mkdir(parents=True, exist_ok=True)
    
    audio_files = sorted(dataset_dir.rglob("MUS/*.wav"))
    
    for audio_path in tqdm(audio_files, desc="MAPS"):
        midi_path = audio_path.with_suffix(".mid")
        if not midi_path.exists():
            midi_path = audio_path.with_suffix(".midi")
        if not midi_path.exists():
            print(f"Skipping {audio_path.name}, no MIDI found")
            continue
        
        try:
            spec = audio_transform(audio_path)
            num_frames = spec.shape[-1]
            frame_labels, onset_labels = midi_labels(midi_path, num_frames)
        except Exception as e:
            print(f"Skipping {audio_path.name}: {e}")
            continue 

        save_name = audio_path.stem + ".pt"
        torch.save({
            "spec": spec,
            "frame_labels": torch.from_numpy(frame_labels),
            "onset_labels": torch.from_numpy(onset_labels),
        }, maps_cache / save_name)


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    cache_dir = Path(cfg.paths.cache_dir) / cfg.audio.name
    audio_transform = AudioTransform(cfg.audio)
    midi_labels = MIDILabels(cfg.audio)
    
    dataset = cfg.get("dataset", "maestro")
    
    if dataset == "maestro":
        preprocess_maestro(cfg, audio_transform, midi_labels, cache_dir)
        
        print("Computing normalizations...")
        train_files = list((cache_dir / "train").glob("*.pt"))
        
        sample = torch.load(train_files[0])["spec"]
        n_bins = sample.shape[0]
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
        print(f"Stats saved to {cache_dir / 'stats.pt'}")
    
    elif dataset == "maps":
        preprocess_maps(cfg, audio_transform, midi_labels, cache_dir)
        print("MAPS preprocessing done.")


if __name__ == "__main__":
    main()