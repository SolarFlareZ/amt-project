
import torch
import torch.nn as nn

from src.models.cnn import ConvBlock


class PianoCRNN(nn.Module):
    def __init__(self, n_bins, num_pitches=88, channels=[64, 128, 256, 512],
                 pool_sizes=[[2,1], [2,1], [2,1], [1,1]], lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 bidirectional=True, use_residual=False, projection_dim=512):
        super().__init__()
        
        
        assert len(pool_sizes) == len(channels), "pool_sizes must match channels length"
        
        # same cnn frontend
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch, pool_size in zip(channels, pool_sizes):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, pool_size=tuple(pool_size), use_residual=use_residual)
            )
            in_ch = out_ch
        
        # cnn ouputs features
        freq_out = n_bins
        for ps in pool_sizes:
            freq_out = freq_out // ps[0]
        cnn_features = channels[-1] * freq_out
        
        # projection, this might cause worse performane, need to keep in mind
        if projection_dim is not None and projection_dim > 0:
            self.projection = nn.Linear(cnn_features, projection_dim)
            lstm_input = projection_dim
        else:
            self.projection = None
            lstm_input = cnn_features
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_out = lstm_hidden * 2 if bidirectional else lstm_hidden
        
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
        
        if self.projection is not None:
            x = torch.relu(self.projection(x))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        
        frame_out = self.frame_head(x)
        onset_out = self.onset_head(x)
        
        return {"frame": frame_out, "onset": onset_out}