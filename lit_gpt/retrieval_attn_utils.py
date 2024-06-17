import os
import torch


def create_dual_triangular_attention_mask(data, eos_id):
    bs, max_seq_length = data.size()
    attention_mask = torch.zeros(bs, max_seq_length, max_seq_length, dtype=torch.float32, device=data.device)

    for batch_idx in range(bs):
        sequence = data[batch_idx]
        # Find the indices of eos tokens
        eos_indices = (sequence == eos_id).nonzero(as_tuple=False).view(-1)
        # Handle cases where the eos_id appears less than 2 times or not at all
        if eos_indices.numel() < 2:
            # If eos_id does not appear or appears only once, fallback to standard lower triangular mask
            # attention_mask[batch_idx, :, :max_seq_length] = torch.tril(torch.ones(max_seq_length, max_seq_length, device=data.device))
            raise ValueError(f"EOS token- {eos_id} does not appear twice in sequence")
        else:
            # Create mask for the first segment (Prefix)
            first_eos_idx = eos_indices[0].item()
            attention_mask[batch_idx, :first_eos_idx+1, :first_eos_idx+1] = torch.tril(torch.ones(first_eos_idx+1, first_eos_idx+1, device=data.device))
            # Create mask for the second segment (Suffix)
            second_eos_idx = eos_indices[1].item()
            attention_mask[batch_idx, first_eos_idx+1:second_eos_idx+1, :second_eos_idx-first_eos_idx] = torch.tril(torch.ones(second_eos_idx-first_eos_idx, second_eos_idx-first_eos_idx, device=data.device))

            # putting True in rest of the indices (Padding locations)
            attention_mask[batch_idx, second_eos_idx+1:, :] = True

    # Reshape the mask to include the additional dimension for heads if necessary
    attention_mask = attention_mask.view(bs, 1, max_seq_length, max_seq_length)
    return attention_mask


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    attn_type="doc_block_attn"):
    """
        Build masks and position id for left to right model.
        Modified from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/utils.py#L162.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    # attention_mask = torch.tril(torch.ones(
    #     (att_mask_batch, seq_length, seq_length), device=data.device)).view(
    #         att_mask_batch, 1, seq_length, seq_length)
    if attn_type == "doc_block_attn":
        attention_mask = create_dual_triangular_attention_mask(data, eod_token)
    if attn_type == "anti_causal_attn":
        attention_mask = torch.triu(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device, dtype=torch.int16)).view(
                att_mask_batch, 1, seq_length, seq_length)


    loss_mask = None
    position_ids = None
    # text = attention_mask[1, 0].tolist()
    # saving text as a txt file
    # with open("attention_mask.txt", "w") as f:
    #     for item in text:
    #         f.write(f"{item}\n")

    # # Loss mask.
    # loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    # if eod_mask_loss:
    #     loss_mask[data == eod_token] = 0.0

    # # Position ids.
    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    # # We need to clone as the ids will be modifed based on batch index.
    # if reset_position_ids:
    #     position_ids = position_ids.clone()

    # if reset_position_ids or reset_attention_mask:
    #     # Loop through the batches:
    #     for b in range(micro_batch_size):

    #         # Find indecies where EOD token is.
    #         eod_index = position_ids[b, data[b] == eod_token]
    #         # Detach indecies from positions if going to modify positions.
    #         if reset_position_ids:
    #             eod_index = eod_index.clone()

    #         # Loop through EOD indecies:
    #         prev_index = 0
    #         for j in range(eod_index.size()[0]):
    #             i = eod_index[j]
    #             # Mask attention loss.
    #             if reset_attention_mask:
    #                 attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
    #             # Reset positions.
    #             if reset_position_ids:
    #                 position_ids[b, (i + 1):] -= (i + 1 - prev_index)
    #                 prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask > 0.5)

    return attention_mask, loss_mask, position_ids