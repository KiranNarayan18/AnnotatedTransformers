import torch

from torch.optim.lr_scheduler import LambdaLR

from models.regularization import LabelSmoothing
from models.model import make_model
from models.config import learning_rate
from models.train import run_epoch
from models.utils import data_generator
from models.loss import SimpleLossCompute, greedy_decode
 

class DummyOptmizer(torch.optim.Optimizer):
  def __init__(self):
    self.param_groups = [{"lr":0}]
    None

  def step(self):
    None

  def zero_grad(self, set_to_none:bool=False) -> None:
    None

class DummyScheduler:
  def step(self):
    None


def example_simple_model():
    V = 11

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
                        model.parameters(), lr=0, betas=(0.9, 0.8), eps=1e-9
                    )
    
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=4000)
    )

    batch_size = 80
    for epoch in range(10):
        model.train()
        run_epoch(
            data_generator(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode='train',
        )

    model.eval()

    run_epoch(
        data_generator(V, batch_size, 5),
        model,
        SimpleLossCompute(model.generator, criterion),
        DummyOptmizer(),  # No need to optimize for evaluation
        DummyScheduler(),
        model='eval'
    )[0]


    model.eval()

    src = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)

    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))



RUN_EXAMPLES = True

def execute_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    fn(*args)



execute_example(example_simple_model)