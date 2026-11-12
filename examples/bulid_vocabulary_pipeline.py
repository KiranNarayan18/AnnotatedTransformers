"""Minimal, dependency-light vocabulary helpers for the examples."""
from transformers import AutoTokenizer

class Vocab(dict):
    """Simple dictionary-backed vocabulary used by the teaching examples."""

    def __init__(self, tokens=None):
        super().__init__()
        if tokens is not None:
            for index, token in enumerate(tokens):
                self[token] = index


# def load_tokenizers():
#     try:
#         spacy_de = spacy.load("de_core_news_sm")
#     except IOError:
#         os.system("python -m spacy download de_core_news_sm")
#         spacy_de = spacy.load("de_core_news_sm")

#     try:
#         spacy_en = spacy.load("en_core_web_sm")
#     except IOError:
#         os.system("python -m spacy download en_core_web_sm")
#         spacy_en = spacy.load("en_core_web_sm")
#     return spacy_de, spacy_en



def load_tokenizers():
    tokenizer_de = AutoTokenizer.from_pretrained(
        "bert-base-german-cased"
    )

    tokenizer_en = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    return tokenizer_de, tokenizer_en


def load_vocab(spacy_de=None, spacy_en=None):
    vocab_src = {token: index for index, token in enumerate(["<blank>", "<s>", "</s>"] + [f"src_{i}" for i in range(32)])}
    vocab_tgt = {token: index for index, token in enumerate(["<blank>", "<s>", "</s>"] + [f"tgt_{i}" for i in range(32)])}
    return vocab_src, vocab_tgt
