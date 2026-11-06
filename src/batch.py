import torch

def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab, device, max_padding=128, pad_id=2):
    bs_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)

    for (_src, _tgt) in batch:

        src_embeddings = torch.tensor(
                                src_vocab(src_pipeline(_src)),
                                dtype=torch.int64,
                                device=device
                                )

        processed_src = torch.cat([bs_id, src_embeddings, eos_id], 0)

        tgt_embeddings = torch.tensor(
                                    tgt_vocab(tgt_pipeline(_tgt)),
                                    dtype=torch.int64,
                                    device=device
                                    )

        processed_tgt = torch.cat([bs_id, tgt_embeddings, eos_id], 0)

        src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id ))
        tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    return (src, tgt)

