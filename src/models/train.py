import time

import torch

from models.config import TrainState
from models.attention import subsequent_mask


class Batch:
    """Prepare source/target tensors and their attention masks."""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            # Teacher forcing: decoder receives all but the final token;
            # labels are the same sequence shifted one position to the left.
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        causal_mask = subsequent_mask(tgt.size(-1)).to(device=tgt.device)
        return tgt_mask & causal_mask


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
):
    if accum_iter < 1:
        raise ValueError("accum_iter must be >= 1")
    if train_state is None:
        train_state = TrainState()

    is_training = mode in ("train", "train+log")
    model.train(is_training)

    start = time.time()
    total_tokens = 0
    total_loss = 0.0
    tokens_since_log = 0

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(data_iter):
        with torch.set_grad_enabled(is_training):
            predictions = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(predictions, batch.tgt_y, batch.ntokens)

            if is_training:
                (loss_node / accum_iter).backward()
                train_state.step += 1
                train_state.samples += batch.src.size(0)
                train_state.tokens += int(batch.ntokens)

                should_step = (i + 1) % accum_iter == 0
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    train_state.accum_step += 1

        batch_tokens = int(batch.ntokens)
        total_loss += float(loss)
        total_tokens += batch_tokens
        tokens_since_log += batch_tokens

        if is_training and mode == "train+log" and i % 40 == 1:
            elapsed = max(time.time() - start, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch Step: {train_state.step}, "
                f"Accum Step: {train_state.accum_step}, "
                f"Tokens: {train_state.tokens}, "
                f"Loss: {float(loss) / max(batch_tokens, 1):.4f}, "
                f"Tokens/sec: {tokens_since_log / elapsed:.2f}, "
                f"Learning Rate: {lr:.6f}"
            )
            start = time.time()
            tokens_since_log = 0

    # Apply a final optimizer update when the epoch ends mid-accumulation.
    if is_training and len(data_iter) % accum_iter:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        train_state.accum_step += 1

    if total_tokens == 0:
        raise ValueError("The data iterator produced no non-padding target tokens")

    return total_loss / total_tokens, train_state
