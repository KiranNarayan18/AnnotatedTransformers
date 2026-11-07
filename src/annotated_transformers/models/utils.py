import copy
import torch
from torch import nn
from ..data.batching import Batch

def clones(module, count):
    """Return independent deep copies of a module."""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))

def data_generator(vocab_size, batch_size, num_batches, sequence_length=10):
    for _ in range(num_batches):
        data = torch.randint(1, vocab_size, (batch_size, sequence_length))
        data[:, 0] = 1
        yield Batch(data, data, pad=0)
