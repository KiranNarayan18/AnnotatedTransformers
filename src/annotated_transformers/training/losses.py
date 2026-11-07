import torch

from ..models.attention import subsequent_mask


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1),
        ) / norm
        return loss.detach() * norm, loss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.full(
        (src.size(0), 1),
        start_symbol,
        dtype=src.dtype,
        device=src.device,
    )

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).to(device=src.device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            prob = model.generator(out[:, -1])
            next_word = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

    return ys
