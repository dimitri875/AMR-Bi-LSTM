import torch
import torch.nn as nn
import torch.nn.functional as F
from .depthPointConv import DepthPointConv
from .residualBlock import ResidualBlock
from .attention import Attention

class AMRModel(nn.Module):
    def __init__(self, num_classes=11,
                 use_attention=True,
                 use_lstm=True,
                 use_residual=True,
                 use_depthwise=True,
                 bidirectional=True):
        super().__init__()

        if use_depthwise:
            self.conv = DepthPointConv(2, 128)
        else:
            self.conv = nn.Conv1d(2, 128, kernel_size=3, padding=1)

        self.use_residual = use_residual
        if use_residual:
            self.resblock = ResidualBlock(128)

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=64,
                batch_first=True,
                bidirectional=bidirectional
            )
            lstm_out = 64 * (2 if bidirectional else 1)
        else:
            lstm_out = 128

        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(lstm_out)

        self.fc = nn.Linear(lstm_out, num_classes)

    def forward(self, x):
        x = self.conv(x)

        if self.use_residual:
            x = self.resblock(x)

        x = self.pool(x)
        x = x.permute(0, 2, 1)

        if self.use_lstm:
            x, _ = self.lstm(x)

        if self.use_attention:
            x = self.attention(x)
        else:
            x = x.mean(dim=1)

        x = self.fc(x)