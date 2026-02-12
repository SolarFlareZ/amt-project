import torch.nn as nn

# maybe have 2 conv layers per block??

# temp change
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(2, 1), use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.pool_size = pool_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)

        if use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.residual_pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)

        if self.use_residual:
            identity = self.residual_conv(identity)
            identity = self.residual_pool(identity)
            out = out + identity

        return out




class PianoCNN(nn.Module):
    def __init__(self, n_bins, num_pitches=88, channels=[64, 128, 256, 512],
                 pool_sizes=[[2,1], [2,1], [2,1], [1,1]], dropout=0.3, use_residual=False):
        super().__init__()
                
        assert len(pool_sizes) == len(channels), "pool_sizes must match channels length"
        
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch, pool_size in zip(channels, pool_sizes):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, pool_size=tuple(pool_size), use_residual=use_residual)
            )
            in_ch = out_ch
        
        freq_out = n_bins
        for ps in pool_sizes:
            freq_out = freq_out // ps[0]
        
        self.fc = nn.Linear(channels[-1] * freq_out, num_pitches)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        for block in self.conv_blocks:
            x = block(x)
        
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x