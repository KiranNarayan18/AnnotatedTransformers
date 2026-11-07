"""Minimal data helpers used by the examples.

The original project keeps the data-loading code outside the package in the
training scripts. The tiny implementation here provides a compatible, 
self-contained batch structure and synthetic dataloaders so the example can run
without external datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.attention import subsequent_mask


@dataclass
class Batch:
    """Prepare source/target tensors and attention masks for a training step."""

    src: torch.Tensor
    tgt: torch.Tensor | None = None
    pad: int = 0

    def __post_init__(self):
        self.src_mask = (self.src != self.pad).unsqueeze(-2)

        if self.tgt is not None:
            self.tgt = self.tgt[:, :-1]
            self.tgt_y = self.tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, self.pad)
            self.ntokens = (self.tgt_y != self.pad).sum()
        else:
            self.tgt_y = None
            self.tgt_mask = None
            self.ntokens = 0

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        causal_mask = subsequent_mask(tgt.size(-1)).to(device=tgt.device)
        return tgt_mask & causal_mask
