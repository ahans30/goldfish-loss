import torch

from torch.utils.data._utils.collate import collate_tensor_fn


def generic_collate_fn(
    batch,
    tokenizer=None,
    block_size=None,
    pad_to_block_size=False,
    add_bos=True,
    add_eos=True,
    collate_checks_enabled=True,
    all_block_size_tensors=False,
):
    if all_block_size_tensors:
        # If we are only dealing with tensors that we _know_ are the same size,
        # we can just use the default collate_tensor_fn
        return collate_tensor_fn(batch)

    if collate_checks_enabled:
        assert isinstance(batch, list), "Batch must be a list."
        type_list = [type(x) for x in batch]
        if str in type_list:
            assert tokenizer is not None, "If batch contains strings, tokenizer must be provided."
            assert tokenizer.pad_id is not None, "Tokenizer must have pad token id since we are dynamically padding."

    # if tokenizer is not None:
    # for now, we assume that if we need it, the tokenizer is always present
    batch = [tokenizer.encode(row, bos=add_bos, eos=add_eos) if type(row) == str else row for row in batch]

    # Now all rows are tokenized
    # logic is a bit generic, could be tightened under encode -> tensor assumption
    if pad_to_block_size:
        batch = [torch.tensor(x[:block_size].tolist() + [tokenizer.pad_id] * (block_size - len(x))) for x in batch]
    else:
        # pad to longest in batch
        max_len = max(len(x) for x in batch)
        batch = [torch.tensor(x.tolist() + [tokenizer.pad_id] * (max_len - len(x))) for x in batch]

    # Now all rows are tensors of the same length.
    # Always slice to block size since the max row length realized could be longer than block size.
    collated_batch = collate_tensor_fn(batch)[:, :block_size]

    # We need to check whether the entire batch consists of padding tokens
    # if so, we raise a StopIteration to signal the exhaustion of all data sources since
    # no real tokens are present in the batch
    if torch.all(collated_batch == tokenizer.pad_id):
        raise StopIteration("All tokens in batch are padding tokens.")

    return collated_batch
