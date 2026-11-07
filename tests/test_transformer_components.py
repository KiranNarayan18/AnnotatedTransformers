import torch
from annotated_transformers.models.attention import subsequent_mask
from annotated_transformers.models.embeddings import PositionalEncoding

def test_subsequent_mask_is_lower_triangular():
    mask = subsequent_mask(4)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4, 4)
    assert torch.equal(mask[0], torch.tril(torch.ones(4, 4, dtype=torch.bool)))

def test_positional_encoding_supports_odd_dimensions():
    layer = PositionalEncoding(d_model=5, dropout=0.0, max_len=8)
    assert layer(torch.zeros(2, 4, 5)).shape == (2, 4, 5)
