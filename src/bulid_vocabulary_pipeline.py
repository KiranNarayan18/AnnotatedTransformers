import os
from collections import Counter

import torch
import spacy
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        spacy.cli.download("de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        spacy.cli.download("en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


class Vocab:
    def __init__(self, stoi, itos, default_idx=3):
        # stoi = String to Integer
        # itos = Integer to String

        self.stoi = stoi
        self.itos = itos

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_idx)

    def __call__(self, token):
        return [self.stoi.get(t, self.default_idx) for t in tokenize]

    def __len__(self):
        return len(self.itos)

    def __get_itos__(self):
        return self.itos

    def __get_stoi__(self):
        return self.stoi
    
    def set_default_index(self, idx):
        self.default_idx = idx


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    ds = load_dataset("bentrevett/multi30k",trust_remote_code=False)
    ds.save_to_disk("./data/dataset/multi30k")


    counter_de = Counter()
    counter_en = Counter()

    for split in ['train', 'validation','test']:
        for ex in ds[split]:
            counter_de.update(tokenize_de(ex['de']))
            counter_en.update(tokenize_en(ex['en']))

    specials = ["<s>", "</s>", "<blank>", "<unk>"]
    # specials = ["<s> : 0", "</s>" : 1, "<blank>". 2, "<unk>" : 3]
    # reserved tokens - these words are included in vocab

    print("Building German Vocabulary..............")
    stoi_de = {w:i for i,w in enumerate(specials)}  ## first adding just specials
    for w, _ in counter_de.most_common():
        if w not in stoi_de:
            stoi_de[w] = len(stoi_de)  ## then adding each unique token and giving an int ## the len of stoi_de keeps increasing so value would be unique
    itos_de = {i:w for w,i in stoi_de.items()}

    print("Building English Vocabulary..............")
    stoi_en = {w:i for i,w in enumerate(specials)}
    for w, _ in counter_en.most_common():
        if w not in stoi_en:
            stoi_en[w] = len(stoi_en)
    itos_en = {i:w for w,i in stoi_en.items()}

    vocab_src = Vocab(stoi_de, itos_de)
    vocab_tgt = Vocab(stoi_en, itos_en)

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not os.path.exists(os.path.join('data', 'models','vocab.pt')):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), os.path.join('data', 'models','vocab.pt'))
    else:
        vocab_src, vocab_tgt = torch.load(os.path.join('data', 'models','vocab.pt'), map_location="cpu", weights_only=False)
    
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if __name__ == '__main__':
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)


