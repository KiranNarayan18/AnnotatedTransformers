"""Synthetic dataloaders used by the educational training example."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from .batching import Batch


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de=None,
    spacy_en=None,
    batch_size=8,
    max_padding=72,
    is_distributed=False,
):
    """Return compact synthetic dataloaders for smoke tests and examples."""
    if is_distributed:
        raise NotImplementedError("Distributed synthetic loaders are not implemented for this lightweight example.")

    seq_len = min(max_padding, 16)
    pad_idx_src = vocab_src.get("<blank>", 0)
    pad_idx_tgt = vocab_tgt.get("<blank>", 0)

    src = torch.randint(3, len(vocab_src), (128, seq_len), device=device)
    tgt = torch.randint(3, len(vocab_tgt), (128, seq_len), device=device)
    src[:, 0] = pad_idx_src
    tgt[:, 0] = pad_idx_tgt

    dataset = TensorDataset(src, tgt)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
