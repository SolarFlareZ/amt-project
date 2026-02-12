import torch


class AddNoise: # adding gaussian noise
    def __init__(self, snr_db=20):
        self.snr_db = snr_db
    
    def __call__(self, spec):
        signal_power = spec.pow(2).mean()
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(spec) * torch.sqrt(noise_power)
        return spec + noise
    


class GainVariation: # should kind of simulate volume change
    def __init__(self, gain_db=-6):
        self.gain = 10 ** (gain_db / 20)
    
    def __call__(self, spec):
        return spec * self.gain