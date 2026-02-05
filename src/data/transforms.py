import torch
import torchaudio
import numpy as np
import librosa

class AudioTransform:
    def __init__(self, cfg):
        self.sample_rate = cfg.sample_rate
        self.cfg = cfg
        self._init_spec_transform(cfg)
    
    def _init_spec_transform(self, cfg):
        if cfg.name == "mel":
            self.spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                f_min=cfg.fmin,
                f_max=cfg.fmax
            )
        elif cfg.name == "stft":
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length
            )
        elif cfg.name == "cqt":
            # Using librosa immedietly in __call__
            self.spec_transform = None
        else:
            raise ValueError(f"Invalid transform: {cfg.name}")

    def _load_audio(self, path):
        waveform, curr_sr = torchaudio.load(path)

        if waveform.shape[0] > 1: # converting to mono
            waveform = waveform.mean(dim=0, keepdim=True)

        if curr_sr != self.sample_rate: # resample
            resampler = torchaudio.transforms.Resample(curr_sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)
    
    def _compute_cqt(self, waveform):
        y = waveform.numpy()
        cqt = librosa.cqt(
            y,
            sr=self.sample_rate,
            hop_length=self.cfg.hop_length,
            n_bins=self.cfg.n_bins,
            bins_per_octave=self.cfg.bins_per_octave,
            fmin=self.cfg.fmin
        )
        return torch.from_numpy(np.abs(cqt)).float()
    
    def __call__(self, audio_path):
        waveform = self._load_audio(audio_path)
        
        if self.cfg.name == "cqt":
            spec = self._compute_cqt(waveform)
        else:
            spec = self.spec_transform(waveform)
        
        return torch.log1p(spec)