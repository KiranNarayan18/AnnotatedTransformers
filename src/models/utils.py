import copy

from torch import nn

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
