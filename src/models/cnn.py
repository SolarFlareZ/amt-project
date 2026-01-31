import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class PianoCNN(nn.Module):
    def __init__(self, n_mels, num_pitches=88, channels=[32, 64, 128, 256], dropout=0.3):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            ConvBlock(1, channels[0], pool_size=(2, 1)),
            ConvBlock(channels[0], channels[1], pool_size=(2, 1)),
            ConvBlock(channels[1], channels[2], pool_size=(2, 1)),
            ConvBlock(channels[2], channels[3], pool_size=(2, 1)),
        ])
        
        # n_mels // 2 // 2 // 2 // 2
        freq_out = n_mels // 16
        self.fc = nn.Linear(channels[3] * freq_out, num_pitches)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: (batch, n_mels, time)
        x = x.unsqueeze(1) # (batch, 1, n_mels, time)
        
        for block in self.conv_blocks:
            x = block(x)
        
        # x: (batch, channels, freq, time)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, -1)  # (batch, time, channels * freq)
        
        x = self.dropout(x)
        x = self.fc(x)  # (batch, time, 88)
        
        return x