class TrainState:
  """Track number of steps, examples and tokens processed"""
  step: int=0
  accum_step: int=0
  samples: int=0
  tokens: int=0


def learning_rate(step, model_size, factor, warmup):
  if step == 0:
    step = 1
  return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))