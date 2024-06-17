"""
This script is helpful to simulate a learning rate schedule if you want to train for an existing hot checkpoint.

By default it configures the hardcoded hyperparameters for the TinyLLaMA model.
https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py#L30:40
You can change the hyperparameters to simulate the learning rate schedule for other models.

TODO: Parameterize the script to accept the hyperparameters as arguments.
"""
import sys
import os
from dataclasses import dataclass
from functools import partial
import torch

# Add the root directory of the project to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pretrain_umd.train import CLISettings, get_lr
from lit_gpt.utils import *

class CfgWithoutValidation(CLISettings):
    def __post_init__(self):
        pass

def main():
    cfg = CfgWithoutValidation(
        max_iters=1_430_512,
        min_lr=4e-5,
        lr_schedule="cosine",
        learning_rate=4e-4,
        warmup_steps=2000
    )
    cfg.warmup_iters = cfg.warmup_steps # assumes steps == iters i.e. gradient accumulation steps = 1

    # Computing hot lr for https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T
    lr = get_lr(it=480_000 + 1, lr_decay_iters=cfg.max_iters, cfg=cfg) # Can use this as new max_lr and resume training
    print(f"Hot LR: {lr}") # Hot LR: 0.00030937179340707335
    return

def tld_loss_sanity():
    cfg = CfgWithoutValidation(
        label_smoothing = 0,
        tld_strategy = 'hash-avalanche',
        k_token_loss_dropout = 3
    )

    vocab_size = 32_000
    block_size = 20
    mbs = 2

    targets_swapped = torch.randint(0, vocab_size, (mbs+2, block_size))
    torch.manual_seed(1337)
    logits = torch.randn(mbs, block_size, vocab_size)
    targets = torch.randint(0, vocab_size, (mbs, block_size))

    swapped_targets = torch.cat((targets_swapped, targets[:1]), dim=0)

    ignore_index = -1

    loss_func = partial(
            chunked_cross_entropy,
            label_smoothing=cfg.label_smoothing,
            tld_strategy=cfg.tld_strategy,
            k_token_loss_dropout=cfg.k_token_loss_dropout,
            tld_start_position=cfg.tld_start_position,
            ignore_index=ignore_index,
    )

    loss = loss_func(logits=logits, targets=targets)
    all_token_loss = loss_func(logits=logits, targets=targets, tld_strategy=None)
    post_tld_targets, _ = apply_tld(targets=targets, strategy=cfg.tld_strategy, k=cfg.k_token_loss_dropout, tld_start_position=cfg.tld_start_position, ignore_index=ignore_index)
    swapped_tld_targets, _ = post_tld_targets, _ = apply_tld(targets=swapped_targets, strategy=cfg.tld_strategy, k=cfg.k_token_loss_dropout, tld_start_position=cfg.tld_start_position, ignore_index=ignore_index)

    assert torch.all(swapped_tld_targets[-1] == post_tld_targets[-1])

    # random TLD strategy
    k = cfg.k_token_loss_dropout

    random_tensor = torch.randint(1, k + 1, size=targets.size())
    mask = (random_tensor == k).int()
    dropped_token_indices = mask.nonzero().reshape(mbs, -1)

    breakpoint()

if __name__ == '__main__':
    # main()
    tld_loss_sanity()
