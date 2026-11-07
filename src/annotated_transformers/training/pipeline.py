
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


from ..models.model import make_model
from .regularization import LabelSmoothing
from ..data.loaders import create_dataloaders
from ..models.config import TrainState, learning_rate
from .engine import run_epoch
from ..data.batching import Batch
from .losses import SimpleLossCompute



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



def train_worker(gpu, ngpus_per_node, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False):
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
    print(f"Train worker process using device: {device}", flush=True)

    if cuda_available:
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt['<blank>']
    d_model = 512
    ## encoderdecoder model
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model = model.to(device)
    module = model
    is_main_process = True
    ## checking for multiple GPUs
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu = 0

    ## regularization
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion = criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config['batch_size'] // ngpus_per_node,
        max_padding=config['max_padding'],
        is_distributed=is_distributed
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate(step, d_model, factor=1, warmup=config['warmup'])
    )
    train_state = TrainState()

    for epoch in range(config['num_epochs']):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if cuda_available:
            try:
                GPUtil.showUtilization()
            except Exception:
                pass
        
        if is_main_process:
            file_path = "%s%.2d.pt" % (config['file_prefix'], epoch)
            torch.save(module.state_dict(), file_path)

        if cuda_available:
            torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptmizer(),
            DummyScheduler(),
            mode='eval',
        )
        print(sloss)
        if cuda_available:
            torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


