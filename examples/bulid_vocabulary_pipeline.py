"""Minimal, dependency-light vocabulary helpers for the examples."""


class Vocab(dict):
    """Simple dictionary-backed vocabulary used by the teaching examples."""

    def __init__(self, tokens=None):
        super().__init__()
        if tokens is not None:
            for index, token in enumerate(tokens):
                self[token] = index


def load_tokenizers():
    return None, None


def load_vocab(spacy_de=None, spacy_en=None):
    vocab_src = {token: index for index, token in enumerate(["<blank>", "<s>", "</s>"] + [f"src_{i}" for i in range(32)])}
    vocab_tgt = {token: index for index, token in enumerate(["<blank>", "<s>", "</s>"] + [f"tgt_{i}" for i in range(32)])}
    return vocab_src, vocab_tgt
