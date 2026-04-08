import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        
        weights = torch.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)        # (batch, hidden_dim)
        
        return context