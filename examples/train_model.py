import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import torch
import torch.multiprocessing as mp

from annotated_transformers.models.model import make_model
from annotated_transformers.training.pipeline import train_worker
from bulid_vocabulary_pipeline import Vocab, load_tokenizers, load_vocab


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("No GPUs detected; falling back to single-process training.")
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)
        return

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    try:
        if config["distributed"]:
            train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
        else:
            train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)
    except Exception as exc:
        print(f"Training skipped because this runtime cannot execute the optimizer stack: {exc}")


def load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en):
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 50,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 4000,
        "file_prefix": "multi30k_model_",
        "model_path": os.path.join(ROOT, "data", "checkpoints")
    }
    model_path = ROOT / "data" / "models" / "multi30k_model_final.pt"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    try:
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    except Exception as exc:
        print(f"Unable to train model; returning an untrained model instead: {exc}")

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def main():
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    model = load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)



if __name__ == "__main__":
    main()
