import copy
import torch
from torch import nn
from models.train import Batch

def clones(module, N):
  "Produce N identical layers"
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def is_interactive_notebook():
  return __name__=="__main__"

def show_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    return fn(*args)

def execute_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    fn(*args)


def data_generator(V, batch_size, nbatches):
  for i in range(nbatches):
    data = torch.randint(1, V, size=(batch_size, 10))
    data[:, 0] = 1
    src = data.requires_grad_(False).clone().detach()
    tgt = data.requires_grad_(False).clone().detach()
    yield Batch(src, tgt, 0)