import torch
from torch import nn


from .utils import clones

## creating mask

def subsequent_mask(size):
    """Mask out subsequent positions for decoder inputs. This is used to prevent the decoder from attending to future tokens in the target sequence during training."""

    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)

    return subsequent_mask


class MultiHeadAttention(nn.Module):
    """ 
    The main idea is that Multi-Head Attention splits the embedding into multiple smaller attention heads,
    runs attention independently in each head, and then combines their results.

    We'll use these example dimensions throughout:

    Batch size = 32
    Sequence length = 10
    Model dimension = 512
    Number of heads = 8
    Dimension per head = 512 / 8 = 64
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h # Number of attention heads inside encoder/decoder layer that is 8
        self.d_k = d_model // h # dimension of model / no of heads -->
                            # if embedding dim is 512 and has 8 heads --> each head will have 64 dims
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        ## why 4 linear layers? --> 3 for query, key, value and 1 for output projection
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0) ## query.shape (32, 10, 512) --> query.size(0) = no of batches

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)for lin, x in zip(self.linears, (query, key, value))            
            ] ## all this to avoid shape mismatch

        ## what is the use of linear is ip and op shape it returns is same
        ## The shape stays the same, but the values change because the layer applies learned weights and biases.

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)) ## contiguous() creates a contiguous copy when needed. This allows the following reshape to work with a straightforward memory layout.
        ## self.h * self.d_k --> 8 * 64 = 512 --> concatinating all the heads back

        del query
        del key
        del value

        return self.linears[-1](x)

