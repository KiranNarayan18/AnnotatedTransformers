class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        ## this would return a boolean matrix of size src tensor, True while the tensor value is != pad val False when value == pad
        ## unsqueeze adds another dim to match std shape transformers process, i.e., [batch_size, number_of_heads, query_length, key_length]

        # [1, 11, 12, 13, 3, 2]
        # 1 - BOS  beginning-of-sequence token
        # 11, 12, 13 = ordinary target tokens.
        # 3 = end-of-sequence token (EOS)
        # 2 = padding token (PAD).

        if tgt is not None:
            self.tgt = tgt[:,:-1] ## removing target column, (all features)
            self.tgt_y = tgt[:,-1] ## target column

            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()  ## how many tokens are present (padding not included)

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words in target."""
        tgt_mask = (tgt != pad).unsqueeze(-2)  ## mask for padding tokens
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)  ## mask for future tokens
        return tgt_mask



def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode='train', accum_iter=1, train_state=TrainState()):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        y_pred = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )

        loss = loss_compute(y_pred, batch.tgt_y, batch.ntokens)

        if mode == 'train' or mode == 'train+log':
            loss.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_accum += 1
                train_state.accum_step += 1
            
            scheduler.step()
        
        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i%40 == 1 and (mode=='train' or mode=='train+log'):
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(f"Epoch Step: {train_state.step}, Accum Step: {train_state.accum_step}, Samples: {train_state.samples}, Tokens: {train_state.tokens}, Loss: {loss.item() / batch.ntokens:.4f}, Tokens per Sec: {tokens / elapsed:.2f}, Learning Rate: {lr:.6f}")
        
        start = time.time()
        tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state



