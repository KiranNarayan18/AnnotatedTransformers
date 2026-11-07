# Annotated Transformers

A from-scratch, modular implementation of the original **Encoder–Decoder Transformer** in PyTorch, based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

This repository is both an implementation and a learning resource: the goal is to make the main Transformer building blocks explicit, understand how they interact, and validate them through small experiments and tests rather than relying on high-level Transformer APIs.

> **Status:** Work in progress. The repository contains an annotated notebook and an evolving modular implementation. Some training and distributed-training paths still require end-to-end validation.

## What is covered

- Token embeddings and sinusoidal positional encodings
- Scaled dot-product attention and multi-head attention
- Encoder self-attention and decoder masked self-attention
- Encoder–decoder (cross) attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Padding and causal attention masks
- Label smoothing and the Transformer warmup learning-rate schedule
- Batch preparation, training utilities, and greedy decoding experiments

## Architecture

The model follows the original encoder–decoder design:

**Encoder:** token embeddings + positional encodings, followed by stacked self-attention and feed-forward sublayers.

**Decoder:** token embeddings + positional encodings, followed by stacked masked self-attention, encoder–decoder cross-attention, and feed-forward sublayers.

Each sublayer uses a residual connection and layer normalization. The decoder's causal mask prevents a position from attending to future target positions during autoregressive training and generation.

## Repository layout

```text
AnnotatedTransformers/
├── notebooks/
│   └── AnnotatedTransformers.ipynb
├── src/
│   ├── annotated_transformers/
│   ├── batch.py
│   ├── bulid_vocabulary_pipeline.py
│   ├── data_loaders.py
│   ├── models/
│   │   ├── attention.py
│   │   ├── config.py
│   │   ├── decoder.py
│   │   ├── embeddings.py
│   │   ├── encoder.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── regularization.py
│   │   ├── train.py
│   │   └── utils.py
│   └── train_pipeline.py
├── tests/
├── pyproject.toml
└── uv.lock
```

The notebook preserves the annotated, exploratory development. The Python modules separate model components and training/data utilities.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PyTorch (CPU or CUDA build appropriate for your machine)

Install the project dependencies:

```bash
git clone https://github.com/KiranNarayan18/AnnotatedTransformers.git
cd AnnotatedTransformers
uv sync
```

Run the test suite:

```bash
uv run pytest
```

> The training pipeline may require additional dataset, tokenizer, language-model and hardware setup. Check the relevant source configuration before launching a full training run.

## Implementation notes

- Attention masks use boolean semantics: **True means an attention position is allowed**.
- Target sequences are shifted for teacher forcing: the decoder receives all tokens except the final token, and labels contain all tokens except the initial token.
- Positional encoding follows the sinusoidal formulation from the paper, using the base 10,000.
- Layer normalization is applied over the final feature dimension.
- Gradient accumulation supports a final incomplete accumulation group.

## Current limitations

This project is an educational implementation, not a production-ready translation system. In particular, full dataset training, distributed execution, checkpoint recovery, and translation quality should be independently validated in the target environment before relying on them.

## References

1. Vaswani et al., [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762).
2. Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
3. [PyTorch documentation](https://pytorch.org/docs/stable/).

## Author

[Kiran Narayan](https://github.com/KiranNarayan18)

[Repository](https://github.com/KiranNarayan18/AnnotatedTransformers)
