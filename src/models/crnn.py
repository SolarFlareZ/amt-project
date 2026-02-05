import torch
import torch.nn as nn

from src.models.cnn import ConvBlock

class PianoCRNN(nn.Module):
    def __init__(self, n_bins, num_pitches=88, channels=[32, 64, 128, 256], 
                 lstm_hidden=256, lstm_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        # CNN
        self.conv_blocks = nn.ModuleList([
            ConvBlock(1, channels[0], pool_size=(2, 1)),
            ConvBlock(channels[0], channels[1], pool_size=(2, 1)),
            ConvBlock(channels[1], channels[2], pool_size=(2, 1)),
            ConvBlock(channels[2], channels[3], pool_size=(2, 1)),
        ])
        
        # CNN output
        freq_out = n_bins // 16
        cnn_features = channels[3] * freq_out
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_out = lstm_hidden * 2 if bidirectional else lstm_hidden
        
        # bilstm outputs
        self.frame_head = nn.Linear(lstm_out, num_pitches)
        self.onset_head = nn.Linear(lstm_out, num_pitches)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        for block in self.conv_blocks:
            x = block(x)
        
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1)
        
        # LSTM
        x, _ = self.lstm(x)
        x = self.dropout(x)
        
        # Dual outputs
        frame_out = self.frame_head(x)
        onset_out = self.onset_head(x)
        
        return {"frame": frame_out, "onset": onset_out}