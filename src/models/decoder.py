from torch import nn
# N = Number of decoder layers stacked in the Transformer (original paper: 6).

# h (heads) = Number of parallel attention heads inside each multi-head attention module (original paper: 8).

from .utils import clones
from .layers import LayerNorm, SublayerConnection

class Decoder(nn.Module):
    """Generic N layer decoder with masking"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self attn, src attn and feed forward """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  ## 3 copies because decoder has three residual connectors (sublayers)

        # deecoder layer processes the input x through three sublayers in order: self-attention, cross-attention, and feed-forward.

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,tgt_mask)) ## why x,x,x ---> query,key,values
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))  ## source attn is cross attn
        ## in cross attention, the key and value come from encoder output(memory), while query is from decoder input (x)
        # src_mask tells the decoder which positions in the encoder's input sequence are padding tokens, so the decoder can ignore those positions during cross-attention.
        return self.sublayer[2](x, self.feed_forward)



        
# Copy              Operation
# self.sublayer[0]  Masked self-attention
# self.sublayer[1]  Cross-attention
# self.sublayer[2]  Feed-forward network


# Masked self-attention Uses tgt_mask
# Cross-attention Uses src_mask


# Decoder sublayer      Attention type          Mask used       Purpose

# 1. Masked multi-head  Self-attention          tgt_mask        Prevents attending to future target tokens and, typically, target padding tokens.   
# attention

#  2. Cross-attention   Encoder–decoder         src_mask        Prevents attending to padding tokens in the encoder's source sequence.
#                      multi-head attention 

# 3. Feed-forward       No attention            No mask         Transforms each token's representation independently.
#   network



