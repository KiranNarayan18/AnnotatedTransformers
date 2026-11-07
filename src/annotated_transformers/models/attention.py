import math

import torch
from torch import nn

from .utils import clones


def subsequent_mask(size: int) -> torch.Tensor:
    """Return a boolean lower-triangular mask allowing each position to see itself and the past."""
    return torch.tril(torch.ones(1, size, size, dtype=torch.bool))


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot-product attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # True means the position is allowed to attend.
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with learned query, key, value and output projections."""

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by the number of heads")

        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
