import torch
import torchaudio

class AudioTransform:
    def __init__(self, cfg):
        self.sample_rate = cfg.sample_rate

        self._init_spec_transform(cfg)
        # might wanna cache resampler for sr, as it is pobably same
    
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
            raise NotImplementedError("cqt is under implementation")
        else:
            raise ValueError(f"invalid transform: {cfg.name}")
        

    def _load_audio(self, path):
        waveform, curr_sr = torchaudio.load(path)

        if waveform.shape[0] > 1: # convert it to mono if it is stereo
            waveform = waveform.mean(dim=0, keepdim=True)

        if curr_sr != self.sample_rate: # resample if sample rate does not match sr from config
            resampler = torchaudio.transforms.Resample(curr_sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)
    
    def __call__(self, audio_path):
        waveform = self._load_audio(audio_path)
        spec = self.spec_transform(waveform)
        return torch.log1p(spec)
