import copy

from torch import nn
from torch.nn.functional import log_softmax, pad

from models.attention import MultiHeadedAttention
from models.layers import PositionwiseFeedForward
from models.encoder import Encoder, EncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.embeddings import Embeddings, PositionalEncoding


class Generator(nn.Module):
  "Define standard linear + softmax generation step"
  def __init__(self, d_model, vocab):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
  """
  A standard Encoder-Decoder architecture. Base for this and many other models
  """
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator


  def forward(self, src, tgt, src_mask, tgt_mask):
    "Take in and process masked src and target sequences."
    encoder_op = self.encode(src, src_mask)
    return self.decode(encoder_op, src_mask, tgt, tgt_mask)

  def encode(self, src, src_mask):
    src_embeddings = self.src_embed(src)
    return self.encoder(src_embeddings, src_mask)

  def decode(self, memory, src_mask, tgt, tgt_mask):
    tgt_embeddings = self.tgt_embed(tgt)
    return self.decoder(tgt_embeddings, memory, src_mask, tgt_mask)


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
    ):

    "Helper: Construct a model from hyperparameters."

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model) ## no of heads, model dimension
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # N no of encoders, 6 as of paper
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) ## initialize the parameters of the model
    
    return model