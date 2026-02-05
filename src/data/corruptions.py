import torch
import torchaudio
import numpy as np


class AddNoise:
    """Add Gaussian noise at specified SNR."""
    def __init__(self, snr_db=20):
        self.snr_db = snr_db
    
    def __call__(self, waveform):
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise


class AddReverb:
    """Add reverb using convolution with impulse response."""
    def __init__(self, ir_path=None, decay=0.5, delay_ms=50, sample_rate=16000):
        if ir_path:
            self.ir, _ = torchaudio.load(ir_path)
        else:
            # Simple synthetic reverb
            delay_samples = int(delay_ms * sample_rate / 1000)
            ir_length = int(sample_rate * 0.5)  # 500ms IR
            ir = torch.zeros(ir_length)
            ir[0] = 1.0
            for i in range(1, ir_length // delay_samples):
                idx = i * delay_samples
                if idx < ir_length:
                    ir[idx] = decay ** i
            self.ir = ir.unsqueeze(0)
    
    def __call__(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        reverbed = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            self.ir.unsqueeze(0).flip(-1),
            padding=self.ir.shape[-1] - 1
        ).squeeze(0)
        
        # Trim to original length
        reverbed = reverbed[:, :waveform.shape[-1]]
        return reverbed.squeeze(0)


class Detune:
    """Pitch shift without changing labels (simulates out-of-tune piano)."""
    def __init__(self, cents=25, sample_rate=16000):
        self.cents = cents
        self.sample_rate = sample_rate
    
    def __call__(self, waveform):
        # cents to semitones
        semitones = self.cents / 100
        
        # Use torchaudio pitch shift
        shifted = torchaudio.functional.pitch_shift(
            waveform.unsqueeze(0),
            self.sample_rate,
            semitones
        )
        return shifted.squeeze(0)