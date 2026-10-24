
import torch

from torch import nn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gemma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (((x-mean) / (std + self.eps)) * self.gemma) + self.beta


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        """add + norm"""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # position-wise feed-forward layer 1
        self.w_2 = nn.Linear(d_ff, d_model)  # position-wise feed-forward layer 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # x = self.w_1(x)
        # x = x.relu()
        # x = self.dropout(x) network with ReLU activation and dropout
        # x = self.w_2(x)

        return self.w_2(self.dropout(self.w_1(x).relu()))