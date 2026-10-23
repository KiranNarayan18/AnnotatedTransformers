import copy

from torch import nn

def clones(module, N):
  "Produce N identical layers"
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



## creating a mask
def subsequent_mask(size):
  """Mask out subsequent positions"""
  attn_shape = (1, size, size)
  subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=-1).type(torch.bool)

  return subsequent_mask == 0