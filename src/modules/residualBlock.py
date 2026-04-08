import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.bn(out)
        
        out += identity
        return F.relu(out)
    