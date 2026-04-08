import torch.nn as nn
import torch.nn.functional as F

class DepthPointConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1, 
            groups=in_channels
        )
        
        self.pointwise = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )
        
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)
    