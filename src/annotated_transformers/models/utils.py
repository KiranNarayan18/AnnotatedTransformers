import copy

import torch
from torch import nn


def _get_batch_cls():
    from ..training.engine import Batch

    return Batch


def clones(module, count):
    """Return independent deep copies of a module."""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))


def data_generator(vocab_size, batch_size, num_batches, sequence_length=10):
    Batch = _get_batch_cls()
    for _ in range(num_batches):
        data = torch.randint(1, vocab_size, (batch_size, sequence_length))
        data[:, 0] = 1
        yield Batch(data, data, pad=0)
