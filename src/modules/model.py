import torch
import torch.nn as nn
import torch.nn.functional as F
from .depthPointConv import DepthPointConv
from .residualBlock import ResidualBlock
from .attention import Attention

class AMRModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Conv part
        self.dsconv = DepthPointConv(2, 128)
        self.resblock = ResidualBlock(128)
        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = Attention(128)  # 64*2 = 128

        # FC
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 2, 128)

        x = self.dsconv(x)         # (batch, 128, 128)
        x = self.resblock(x)       # (batch, 128, 128)
        x = self.pool(x)           # (batch, 128, 64)

        # Prepare for LSTM → (batch, seq_len, features)
        x = x.permute(0, 2, 1)     # (batch, 64, 128)

        x, _ = self.lstm(x)        # (batch, 64, 128)

        x = self.attention(x)      # (batch, 128)

        x = self.fc(x)             # (batch, num_classes)

        return x