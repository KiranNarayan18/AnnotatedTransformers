from datasets import load_dataset

import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP


from bulid_vocabulary_pipeline import tokenize
from batch import collate_batch
from models.config import TrainState
from models.loss import SimpleLossCompute
from models.model import make_model
from models.train import run_epoch



class Multi30kDataset(Dataset):
    """Map-style wrapper for HF Multi30k dataset."""

    def __init__(self, split):
        ds = load_dataset("./data/dataset/multi30k", split=split, trust_remote_code=False)
        self.data = list(ds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx]['de'], self.data[idx]['en'])


def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=12000, max_padding=128, is_distributed=True):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.__get_stoi__()["<blank>"],

        )

    
    train_dataset = Multi30kDataset("train")
    valid_dataset = Multi30kDataset("validation")

    train_sampler = (DistributedSampler(train_dataset) if is_distributed else None)
    valid_sampler = (DistributedSampler(valid_dataset) if is_distributed else None)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        collate_fn=collate_fn
    )

    return train_dataloader, valid_dataloader


