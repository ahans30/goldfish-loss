"""
This script is originally adapted from and inspired by the tinyllama.py and 
redpajama.py scripts in the lit-gpt/pretrain directory.

The lit-gpt authors designed this such that setup -> train reads ~linearly.
"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Union, Optional, Any
import json
import random

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path = [str(wd)] + sys.path

from axonn_fabric.fabric import AxoNNFabric

from lit_gpt.model import Block, CausalSelfAttention, LLaMAMLP
from lit_gpt.model import GPT, Config
from lit_gpt.model_axonn import GPT as GPT_axonn
from lit_gpt.model_axonn import Config as Config_axonn

from lit_gpt.tokenizer import Tokenizer
from lit_gpt.packed_cycle_dataset import CombinedDataset, PackedDataset
from lit_gpt.huggingface_dataset import HuggingfaceDataset
from lit_gpt.data_loading_utils import generic_collate_fn
from lit_gpt.utils import (
    chunked_cross_entropy,
    apply_goldfish,
    num_parameters,
    check_valid_checkpoint_dir,
    load_checkpoint,
)
from lit_gpt.data_scheduler_utils import DataSchedulerTracker, DataScheduler


import logging

stdout_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dataclasses import dataclass, field, asdict
from jsonargparse import CLI
import re
import re

end_time = time.time()
stdout_log.info(f"Time to load libraries: {end_time - global_start_time:.02f} seconds.")

####################################################################################################
# Setup functions.
####################################################################################################


@dataclass
class CLISettings:
    # Main settings
    run_name: str = "default-run"  # The name for logging.
    out_dir: str = None  # The directory to save checkpoints. Required.
    resume: bool = True  # Whether to resume from a checkpoint in the out_dir.
    max_tokens: Optional[int] = 1000000000000  # The maximum number of tokens to train on (determines max_iters).
    max_iters: Optional[int] = None
    seed: int = 1337  # The random seed to use for reproducibility.

    # Model configuration
    model_name: str = "tiny-llama-1.1b"  # The model name to use when creating the model from config.py
    block_size: int = 2048  # The block size to use (lit-gpt-ese for sequence length).
    ignore_block_size_mismatch: bool = False  # Whether to ignore block size mismatch.
    model_checkpoint: Optional[str] = None  # The model checkpoint to load. Else, from config.
    doc_block_attn: bool = False  # Whether to mask out the attention between tokens from different documents.
    cache_attn: bool = False  # Whether to train the model with cache attention with cache tokens randomly inserted.
    eod_token: Optional[str] = None  # 'eos','bos','pad' The end-of-document token name (used for doc-block-attn).

    # Training hyperparameters
    world_batch_size: int = 2048  # The total batch size across all devices and nodes.
    learning_rate: float = 0.0004  # The learning rate.
    warmup_steps: int = 2000  # The number of warmup steps.
    weight_decay: float = 0.1  # The weight decay.
    beta1: float = 0.9  # The beta1 parameter for the Adam optimizer.
    beta2: float = 0.95  # The beta2 parameter for the Adam optimizer.
    adamw_eps: float = 1e-8  # The eps parameter for the Adam optimizer
    grad_clip: float = 1.0  # The gradient clipping value.
    lr_schedule: str = "cosine"  # The learning rate schedule to use.
    decay_lr: bool = True  # Whether to decay the learning rate.
    min_lr: float = 0.00004  # The minimum learning rate to decay to.
    no_weight_decay_for_bias_and_norm_params: bool = False  # do not use weight decay for bias and norm params

    # Regularization
    neptune_from_tokens: Optional[int] = None  # Tokens from which NEPTune mode is activated. Set `None` for no noise.
    neptune_till_tokens: Optional[int] = None  # Tokens until which NEPTune mode is activated.
    neptune_noise_alpha: Optional[float] = None  # NEPTune noise alpha. If set to 0 or None, no noise will be added.
    label_smoothing: float = 0.0  # The label smoothing to use.
    goldfish_strategy: Union[None, str] = None  # The strategy to use for goldfish. Set `None` for no goldfish. Options: 'static'
    k_goldfish: Union[None, int] = (
        None  # goldfish k. Every k-th token will be dropped from the loss computation (`None` for no goldfish).
    )
    goldfish_start_position: int = 0  # goldfish start position. Start dropping tokens from this position.
    goldfish_context_width: int = 4  # goldfish context width. Only for 'hash-table' strategy.

    # Implementation and backend
    fabric_strategy: str = "ddp"  # The fabric strategy to use: ddp, fsdp, axonn_tp.
    fabric_precision: str = "bf16-true"  # The precision to use for the fabric.
    micro_batch_size: int = 4  # The micro batch size to use.
    compile_model: bool = True  # Whether to compile the model.
    matmul_precision: str = "high"  # The matmul precision to use, not from original arguments, add if necessary.
    dataloader_num_workers: int = 0  # The number of workers to use for the dataloaders.
    n_chunks: int = 4  # The number of chunks to preload at a time from packed dataset.
    tensor_parallel_size: int = 1  # The size of the tensor parallel dimension.
    torch_dist_init_barrier: bool = False  # Whether to make torch.distributed.init_process_group() blocking.
    gradient_checkpointing_axonn: bool = False  # Whether to use activation checkpointing in AxoNN.

    # Logging
    logger_name: str = "wandb"  # The logger to use for logging, only supports "wandb" for now.
    logger_project: str = "tinyllama"  # The logger/wandb project to log to.
    data_telemetry: bool = False  # Data telemetry switch, set based on needs.
    shape_watching_iters: int = 3  # Number of iterations to watch shapes for. Set to 0 to disable.
    log_step_interval: int = 1  # The base interval for logging (scales with gradient_accumulation_steps).
    eval_iters: int = 100  # The number of iterations to process during a validation loop.
    save_and_eval_interval: int = 2000  # The number of iterations between saving and evaluating.
    save_step_interval: int = None  # The number of iterations between saving.
    eval_step_interval: int = None  # The number of iterations between evaluating.
    save_last_step: bool = False  # Whether to save the checkpoint at the last step
    save_n_min_before_job_done: int = None  # Save the checkpoint n minutes before current job done
    sanity_validate: bool = True  # Whether to run a short sanity check validation loop at the start.
    measure_flops: bool = False  # Whether to measure the flops of the model.
    torch_cpp_log_level: str = None  # log level for the torch C++ backend, ie. elevated would be "INFO"
    torch_distributed_debug: str = None  # log debug level for torch distributed, ie. elevated would be "DETAIL"

    # Data Handling
    text_key: str = "text"  # The default key to use for the text field in the dataset (HFDS only).
    pad_to_block_size: bool = False  # Whether to pad to the block size (HFDS only).
    add_bos: bool = True  # Whether to add the BOS token to the input (HFDS only).
    add_eos: bool = True  # Whether to add the EOS token to the input (HFDS only).
    shuffle_filenames: bool = True  # Shuffle filenames glob'd up for each prefix before creating the datasets.
    collate_checks_enabled: bool = True  # Enable checks for the collate function.
    all_block_size_tensors: bool = False  # Assume all datasets return tensors with the same size, may reduce latency.
    data_config: Union[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "train_data": [
                {
                    "type": "pkds",
                    "prefix": "",
                    "weight": 1,
                },
            ],
            "val_data": [
                {
                    "type": "pkds",
                    "prefix": "",
                    "weight": 1,
                },
            ],
        }
    )
    # The directories containing the training/validation data.
    train_data_dir: str = (
        "/lustre/orion/csc569/proj-shared/language_datasets/processed/spj_star_combined_full_tinyllama_tokd"
    )
    val_data_dir: str = (
        "/lustre/orion/csc569/proj-shared/language_datasets/processed/spj_star_combined_full_tinyllama_tokd"
    )
    # The path to the tokenizer to use [required to identify pad_token_id even for pkds]
    tokenizer_path: str = (
        "/lustre/orion/csc569/proj-shared/language_models/external/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )

    def __post_init__(self):
        # Validate arguments
        if not self.out_dir:
            raise ValueError("out_dir must be specified.")

        # If data_config is a string, load it from a file.
        if isinstance(self.data_config, str):
            try:
                with open(self.data_config) as json_file:
                    self.data_config = json.load(json_file)
            except Exception as e:
                raise ValueError(
                    f"data_config passed was a string, but failed to load as a json object from {self.data_config}: {e}"
                )

        # Tensor parallelism is implemented by the AxoNN fabric only.
        if self.tensor_parallel_size > 1:
            assert self.fabric_strategy == "axonn_tp", "tensor_parallel_size > 1 implies use of axonn_tp."

        # Set the various log levels for torch.
        if self.torch_cpp_log_level:
            os.environ["TORCH_CPP_LOG_LEVEL"] = self.torch_cpp_log_level
        if self.torch_distributed_debug:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = self.torch_distributed_debug

        # Init barrier
        if self.torch_dist_init_barrier:
            os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        env_vars = {"TORCH_DIST_INIT_BARRIER": os.getenv("TORCH_DIST_INIT_BARRIER", "0")}

        # Parse env variables into additional cfg
        os.environ["RANK"] = str(os.getenv("SLURM_PROCID", "0"))
        env_vars.update(
            {
                "SLURM_JOB_ID": int(os.getenv("SLURM_JOB_ID", 0)),
                "SLURM_ARRAY_JOB_ID": int(os.getenv("SLURM_ARRAY_JOB_ID", 0)),
                "SLURM_ARRAY_TASK_ID": int(os.getenv("SLURM_ARRAY_TASK_ID", 0)),
                "SLURM_ARRAY_TASK_COUNT": int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1)),
                "MASTER_ADDR": os.getenv("MASTER_ADDR", 0),
                "MASTER_PORT": int(os.getenv("MASTER_PORT", 0)),
                "WORLD_SIZE": int(os.getenv("WORLD_SIZE", 1)),
                "RANK": int(os.getenv("SLURM_PROCID", "0")),
                # more robust than cuda.device_count() in case of gpu isolation in SLURM:
                "devices": int(os.getenv("SLURM_NTASKS_PER_NODE", 1)),
                "num_nodes": int(os.getenv("SLURM_JOB_NUM_NODES", 1)),
            }
        )
        self.__dict__.update(env_vars)
        # SLURM_NTASKS_PER_NODE is not set if you don't pass it to srun
        # this solution is more general.
        self.devices = divide(int(os.getenv("SLURM_NTASKS", 1)), self.num_nodes)

        # Add any derived cfg here
        self.node_batch_size = self.world_batch_size // self.num_nodes

        if self.save_step_interval is None:
            self.save_step_interval = self.save_and_eval_interval
        if self.eval_step_interval is None:
            self.eval_step_interval = self.save_and_eval_interval

        self.batch_size = self.node_batch_size // self.devices
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        self.warmup_iters = self.warmup_steps * self.gradient_accumulation_steps
        self.log_iter_interval = self.log_step_interval * self.gradient_accumulation_steps
        self.dataset_names = [i["name"] if "name" in i else i["prefix"] for i in self.data_config["train_data"]]

        assert len(set(self.dataset_names)) == len(
            self.data_config["train_data"]
        ), "please provide different names for each subset"

        # Any additional sanity checks here.
        assert self.gradient_accumulation_steps > 0, "derived gradient_accumulation_steps must be > 0"
        assert (
            self.world_batch_size
            == self.micro_batch_size * self.gradient_accumulation_steps * self.devices * self.num_nodes
        ), "world batch size should be: micro_batch_size * gradient_accumulation_steps * devices * num_nodes"
        self._validate_regularization()

        # cache_attn and doc_block_attn checks
        if self.cache_attn:
            raise NotImplementedError("cache_attn is not implemented yet.")

        if self.doc_block_attn and self.fabric_strategy == "axonn_tp":
            raise NotImplementedError("doc_block_attn is not implemented for AxoNN TP model yet.")

        if self.doc_block_attn and self.eod_token is None:
            raise ValueError("doc_block_attn requires eod_token to be set.")

    def _validate_regularization(self):
        # NEPTune specific sanity checks
        if self.neptune_noise_alpha:
            # If non-default neptune alpha specified but not start and end behavior
            assert (
                self.neptune_from_tokens is not None or self.neptune_till_tokens is not None
            ), "Expected either of \
            `neptune_from_tokens` or `neptune_from_tokens` to be specified for NEPTune use. Are you sure you want use \
            NEPTune? Please set `neptune_noise_alpha` to `None` (default) if not."
        if self.neptune_from_tokens is not None or self.neptune_till_tokens is not None:
            # If non-default neptune start/end behavior specified but not neptune alpha
            assert (
                self.neptune_noise_alpha
            ), f"no noise will be added regardless of \
            neptune_from_tokens={self.neptune_from_tokens} or neptune_till_tokens={self.neptune_till_tokens} \
            because neptune_noise_alpha={self.neptune_noise_alpha} if falsy. Are you sure you want to use NEPTune? Set \
            both `neptune_from_tokens` and `neptune_till_tokens` to `None` (default) if not."
            # Assertions when both start and end NEPTune behavior is passed.
            if self.neptune_from_tokens is not None and self.neptune_till_tokens is not None:
                assert (
                    self.neptune_from_tokens >= self.neptune_till_tokens
                ), f"Inconsistent / contradictory NEPTune config \
                passed. Currently, `neptune_from_tokens` >= `neptune_till_tokens`. NEPTune mode cannot both start from \
                {self.neptune_from_tokens} tokens and end after {self.neptune_till_tokens} tokens."

        # Derived cfg for NEPTune in case defaults are used.
        if self.neptune_noise_alpha:
            # set NEPTune defaults for (neptune_from_tokens, neptune_till_tokens) to (0, max_tokens) if xor set to None
            self.neptune_from_tokens = self.neptune_from_tokens or 0
            self.neptune_till_tokens = self.neptune_till_tokens or self.max_tokens

        # goldfish sanity check
        assert not ((self.goldfish_strategy is None) ^ (self.k_goldfish is None)), \
            "both goldfish param must be set or None"
        if self.k_goldfish is not None:
            assert self.k_goldfish >= 2, "k_goldfish must be >= 2 or None (for no goldfish)"

        assert (self.max_iters is not None) ^ (
            self.max_tokens is not None
        ), f"only max_iters ({self.max_iters}) xor max_tokens ({self.max_tokens}) can be specified"


def divide(a, b):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    # Do they AMD cards pick up on any of this? :
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway


def setup_fabric(cfg: CLISettings) -> Tuple[L.Fabric, Union[Logger, WandbLogger]]:
    """Sets up the fabric and logger based on the cfg."""
    set_torch_flags(cfg)  # should come before fabric setup

    # Instantiate the logger.
    if cfg.logger_name == "wandb":
        logger = WandbLogger(project=cfg.logger_project, name=cfg.run_name, save_dir=cfg.out_dir)
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option.")

    # Instantiate the fabric.
    if cfg.fabric_strategy == "axonn_tp":
        fabric = AxoNNFabric(tensor_parallel_grid=[1, 1, cfg.tensor_parallel_size], loggers=[logger])
        fabric.print(f"> global_batch_size = {cfg.world_batch_size}")
        fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
        fabric.print(
            f"> global_world_size_for_creating_dataloader = {fabric.global_world_size_for_creating_dataloader}"
        )
        batch_size_per_gpu = divide(cfg.world_batch_size, fabric.global_world_size_for_creating_dataloader)
        cfg.micro_batch_size = divide(batch_size_per_gpu, cfg.gradient_accumulation_steps)
        # cfg.dtype = torch.bfloat16 # @Prajwal/@Siddharth: Do we use this? it's not json serializable.

        fabric.print(f"> global_batch_size = {cfg.world_batch_size}")
        fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
        fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")

        fabric.print(f"Using AxoNNFabric with tensor parallelism = 1x1x{cfg.tensor_parallel_size}")
        fabric.launch()
    else:
        if cfg.fabric_strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, activation_checkpointing_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
        elif cfg.fabric_strategy == "ddp":
            strategy = DDPStrategy()
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        # Instantiate and launch/initialize the fabric distributed environment management.
        fabric = L.Fabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=[logger],
            num_nodes=cfg.num_nodes,
        )
        fabric.print(f"Using Lightning Fabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

    # Now we call the main function with the fabric and cfg.
    main(fabric, cfg)


####################################################################################################
# Main driver functions.
####################################################################################################


def main(fabric: L.Fabric, cfg: CLISettings):
    """The main driver function for the training script."""
    start_time = time.time()

    # Get job remainin time
    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            try:
                global_total_time = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()
                global_total_time = global_total_time.strip("\n")
                global_total_time = [int(i) for i in re.split(":|-", global_total_time)]
                if len(global_total_time) == 4:
                    global_total_time = (
                        24 * 3600 * global_total_time[0]
                        + 3600 * global_total_time[1]
                        + 60 * global_total_time[2]
                        + global_total_time[3]
                    )
                elif len(global_total_time) == 3:
                    global_total_time = 3600 * global_total_time[0] + 60 * global_total_time[1] + global_total_time[2]
                elif len(global_total_time) == 2:
                    global_total_time = 60 * global_total_time[0] + global_total_time[1]
            except:
                global_total_time = 9999999999999999
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = None

        global_total_time = fabric.broadcast(global_total_time, 0)
        cfg.global_total_time = global_total_time

    # Set up the dataloaders and model.
    if cfg.fabric_strategy == "axonn_tp":
        model_config = Config_axonn.from_name(cfg.model_name)
    else:
        model_config = Config.from_name(cfg.model_name)

    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        # Last step before we move on is to dump the cfg to a file in the out_dir.
        # This is is itself loadable as a config by passing like train.py --config run_config.json
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(model_config), f, indent=4)

    # Add cache token as special token and update the model config.
    vocab_size = model_config.vocab_size
    if cfg.cache_attn:
        # TODO: might want to pad to a power of 2 to be optimal on hardware.
        # TODO: to add multiple cache tokens with distinct tokens ids.
        model_config.padded_vocab_size = vocab_size + 1
    if cfg.tokenizer_path:
        tokenizer = Tokenizer(cfg.tokenizer_path)
        if tokenizer.pad_id is None:
            tokenizer.pad_id = -1
    else:
        tokenizer = None

    # Set the EOD token id for doc-block-attn.
    cfg.eod_token_id = None
    if cfg.doc_block_attn:
        assert tokenizer is not None, "Tokenizer must be provided to set eod_token_id."
        if cfg.eod_token == "eos":
            cfg.eod_token_id = tokenizer.eos_id
        elif cfg.eod_token == "bos":
            cfg.eod_token_id = tokenizer.bos_id
        elif cfg.eod_token == "pad":
            cfg.eod_token_id = tokenizer.pad_id
        assert cfg.eod_token_id is not None, "Requested eod_token not found in tokenizer."

    # set NEPTune config and print behavior to be expected for this run.
    model_config.neptune_noise_alpha = cfg.neptune_noise_alpha
    if cfg.neptune_noise_alpha:
        start_from = "start of training" if cfg.neptune_from_tokens == 0 else str(cfg.neptune_from_tokens) + " tokens"
        end_on = (
            "end of training."
            if cfg.neptune_till_tokens == cfg.max_tokens
            else str(cfg.neptune_from_tokens) + " tokens are seen."
        )
        fabric.print(f"NEPTune will be used from {start_from} till {end_on}")
    else:
        fabric.print("NEPTune is NOT used for this run.")

    if cfg.k_goldfish is not None:
        fabric.print(f"goldfish will be used with k={cfg.k_goldfish} with strategy {cfg.goldfish_strategy}.")
        if "hash" in cfg.goldfish_strategy:
            fabric.print(f"goldfish context width is {cfg.goldfish_context_width}")
        fabric.print(f"Every {cfg.k_goldfish}-th token will be dropped from loss")
        fabric.print(f"goldfish will start from position {cfg.goldfish_start_position} in training sequence.")
    else:
        fabric.print("goldfish is NOT used for this run.")

    t0 = time.time()
    # On block size, moved this here to be more explicit that this is happening ...
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == model_config.block_size, "cfg.block_size must match config.block_size"
    # Increase by one to actually be supervising "block_size" tokens in every update after rshift.
    cfg.loader_block_size = cfg.block_size + 1

    train_dataloader, val_dataloader, data_scheduler_tracker = create_dataloaders(
        batch_size=cfg.micro_batch_size,
        block_size=cfg.loader_block_size,
        fabric=fabric,
        seed=(
            cfg.seed
            + (fabric.global_rank_for_creating_dataloader if cfg.fabric_strategy == "axonn_tp" else fabric.global_rank)
        ),
        cfg=cfg,
        tokenizer=tokenizer,
    )

    if cfg.fabric_strategy == "axonn_tp":
        # TODO: @Prajwal/@Siddharth Make this API the same as lightning fabric
        fabric.setup_dataloaders(train_dataloader, val_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.print(f"Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)

    if cfg.model_checkpoint is not None:
        check_valid_checkpoint_dir(Path(cfg.model_checkpoint))

    fabric.print(f"Loading model with {model_config.__dict__}")
    t0 = time.time()

    if cfg.fabric_strategy == "axonn_tp":
        model = GPT_axonn(model_config, gradient_checkpointing=cfg.gradient_checkpointing_axonn)
        model.apply(partial(init_weights, n_layer=model_config.n_layer, n_embd=model_config.n_embd, axonn_tp=True))
    else:
        with fabric.init_module(empty_init=False):
            # end-of-document token id (used for doc-block-attn)
            model = GPT(model_config, eod_token=cfg.eod_token_id)
            model.apply(partial(init_weights, n_layer=model_config.n_layer, n_embd=model_config.n_embd))

    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # With fabric and the model up, we can compute a few last derived cfg.

    if cfg.max_iters is None:
        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_iter = cfg.micro_batch_size * cfg.block_size
        cfg.max_iters = cfg.max_tokens_per_device // cfg.tokens_per_iter

    data_scheduler = DataScheduler(data_scheduler_tracker, cfg.data_config["train_data"], cfg)
    data_scheduler.step(0)

    # Report the full cfg set for the run.
    if fabric.global_rank == 0:
        fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, indent=4)}")

    if cfg.logger_name in ("tensorboard", "wandb"):
        if cfg.fabric_strategy == "axonn_tp":
            # Only log from rank 0 for the logger on axonn
            if fabric.global_rank == 0:
                fabric.logger.log_hyperparams(cfg.__dict__)
        else:
            # L.Fabric only logs this from rank 0
            fabric.logger.log_hyperparams(cfg.__dict__)

    # Compile the model.
    # FIXME, this is only applicable if batch size is fixed. In future, if using
    # a dataset with variable length sequences, should not allow this.
    # Also, I think this timing is inaccurate...
    if cfg.compile_model:
        t0 = time.time()
        model = torch.compile(model)
        fabric.print(f"Time to compile model: {time.time() - t0:.02f} seconds.")

    t0 = time.time()

    model = fabric.setup(model)

    fabric.print(f"Time to setup model: {time.time() - t0:.02f} seconds.")

    t0 = time.time()
    if cfg.no_weight_decay_for_bias_and_norm_params:
        wd_params = []
        no_wd_params = []

        for name, param in model.named_parameters():
            no_wd = "norm" in name or "bias" in name
            if no_wd:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        param_groups = []
        if wd_params:
            param_groups.append({"params": wd_params})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            foreach=False,
            eps=cfg.adamw_eps,
        )

    else:
        # Set up the optimizer and training state object.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            foreach=False,
            eps=cfg.adamw_eps,
        )
    optimizer = fabric.setup_optimizers(optimizer)

    fabric.print(f"Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        # "cfg": cfg,
        "iter_num": 0,
        "step_count": 0,
    }

    t0 = time.time()
    # If resuming, determine the checkpoint to resume from.
    resume_ckpt = None
    if cfg.resume is True:
        if cfg.fabric_strategy == "axonn_tp":
            base_chkpt_path = Path(cfg.out_dir) / fabric.get_tensor_parallel_prefix_for_checkpoint()
        else:
            base_chkpt_path = Path(cfg.out_dir)
        ckpt_paths = list(base_chkpt_path.glob(f"*-{cfg.run_name}.pth"))
        if len(ckpt_paths) > 0:
            resume_ckpt = max(
                ckpt_paths,
                key=(lambda p: int(p.name.split("-")[1].split(f"-{cfg.run_name}.pth")[0])),
            )
            fabric.print(f"Resuming training from {resume_ckpt}")
            fabric.load(resume_ckpt, state)

    if resume_ckpt is None and cfg.model_checkpoint is not None:
        checkpoint_path = f"{cfg.model_checkpoint}/lit_model.pth"
        fabric.print(f"Loading pretrained model checkpoint from {checkpoint_path}")
        load_checkpoint(fabric, state["model"], checkpoint_path)

    fabric.print(f"Time to load model checkpoint: {time.time() - t0:.02f} seconds.")

    end_time = time.time()
    stdout_log.info(f"Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    # Now we call the train function with the fabric, state, and dataloaders.
    train_time = time.time()
    train(
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        resume_path=resume_ckpt,
        cfg=cfg,
        data_scheduler=data_scheduler,
    )
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, resume_path=None, cfg=None, data_scheduler=None):
    """The main training loop."""

    # Do some checks on the val loop and the throughput of the model.
    model = state["model"]
    optimizer = state["optimizer"]
    tokenizer = state["tokenizer"]

    if cfg.fabric_strategy == "axonn_tp":
        cache_token_id = fabric.unwrapped_model.config.vocab_size
    else:
        cache_token_id = model.config.vocab_size

    if cfg.sanity_validate:
        validate(fabric, model, val_dataloader, max_iters=2, cfg=cfg, tokenizer=tokenizer)  # sanity check

    if cfg.fabric_strategy == "axonn_tp":
        throughput = None
    else:
        throughput = ThroughputMonitor(fabric, window_size=5)

    if cfg.measure_flops:
        with torch.device("meta"):
            meta_model = GPT(model.config)
            x = torch.randint(0, 1, (cfg.micro_batch_size, cfg.block_size))
            model_fwd = lambda: meta_model(x)
            model_loss = lambda y: chunked_cross_entropy(
                y, x, chunk_size=0,
                label_smoothing=cfg.label_smoothing,
                k_goldfish=cfg.k_goldfish,
                goldfish_start_position=cfg.goldfish_start_position,
                goldfish_context_width=cfg.goldfish_context_width
            )
            measured_flops = measure_flops(meta_model, model_fwd, model_loss)
            fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
            cfg.measured_flops = measured_flops
            del meta_model, x
    else:
        cfg.measured_flops = -1.0  # to be clear that we didn't measure flops

    initial_iter = state["iter_num"]
    train_iterator = iter(train_dataloader)

    # Resume data loader state by fast-forwarding through all seen batches.
    # If we migrate to the streaming dataset in future, we might not need this.
    if resume_path:
        resume_t0 = time.time()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")

            data_scheduler.step(resume_iter + 1)

        fabric.barrier()
        fabric.print(f"Resuming data loader finished. Took {time.time() - resume_t0:.1f} seconds to reach iteration")

    # Set up global loss monitor.
    running_loss = RunningMean(window=cfg.gradient_accumulation_steps, sync_on_compute=False).to(fabric.device)
    # Below are used for goldfish jobs only.
    running_all_token_loss = RunningMean(window=cfg.gradient_accumulation_steps, sync_on_compute=False).to(fabric.device)
    running_dropped_tokens_loss = RunningMean(
                                        window=cfg.gradient_accumulation_steps, 
                                        sync_on_compute=False
                                    ).to(fabric.device)

    if tokenizer:
        loss_func = partial(
            chunked_cross_entropy,
            label_smoothing=cfg.label_smoothing,
            ignore_index=tokenizer.pad_id,
            goldfish_strategy=cfg.goldfish_strategy,
            k_goldfish=cfg.k_goldfish,
            goldfish_start_position=cfg.goldfish_start_position,
            goldfish_context_width=cfg.goldfish_context_width,
        )
    else:
        loss_func = partial(
            chunked_cross_entropy,
            label_smoothing=cfg.label_smoothing,
            goldfish_strategy=cfg.goldfish_strategy,
            k_goldfish=cfg.k_goldfish,
            goldfish_start_position=cfg.goldfish_start_position,
            goldfish_context_width=cfg.goldfish_context_width,
        )

    fabric.barrier()
    total_t0 = time.time()

    # Determine and set the learning rate for initial step.
    # FIXME the lr schedule is computed as a function of iters not optim steps.
    # These are different if gradient_accumulation_steps > 1.
    # There doesn't seem to be anything _incorrect_ about this, but it might
    # not be very intuitive when picking schedule params.
    lr = get_lr(it=state["iter_num"], lr_decay_iters=cfg.max_iters, cfg=cfg) if cfg.decay_lr else cfg.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    have_not_saved = True
    # Main training loop.
    for train_data in train_iterator:
        if state["iter_num"] >= cfg.max_iters:
            break

        # switch NEPTune on/off based on config before iter++
        if cfg.neptune_noise_alpha:
            total_tokens_seen = state["iter_num"] * cfg.micro_batch_size * cfg.block_size * fabric.world_size
            if cfg.neptune_from_tokens <= total_tokens_seen <= cfg.neptune_till_tokens:
                model.config.neptune_noise_alpha = cfg.neptune_noise_alpha
            else:
                model.config.neptune_noise_alpha = None

        state["iter_num"] += 1

        # NOTE, this might be the one time we could use time.perf_counter() instead of time.time()
        # Generally for > sec resolution timing, .time() is sufficient and less expensive.
        # https://superfastpython.com/time-time-vs-time-perf_counter
        iter_t0 = time.time()

        # Realize the input and target tensors.
        bsz, seq_len = train_data.shape
        input_ids = train_data[:, 0 : (seq_len - 1)].contiguous().long()
        if cfg.fabric_strategy == "axonn_tp":
            input_ids = input_ids.cuda()
        # for the input we need to replace any pad ids with the eos token
        # knowing that they're trailing so they wont contrib to activations
        # but that they do need to be valid indices in the model's embedding layer
        if tokenizer:
            input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id
        # Note that we are _not_ doing this operation for the targets,
        # since this is where we actually need the pad tokens to be present for loss to ignore them.

        targets = train_data[:, 1:(seq_len)].contiguous().long()

        # Insert random cache tokens here.
        # FIXME: make the probability configurable, currently hardcoded as 0.5.
        if cfg.cache_attn and torch.rand(1) > 0.5:
            cache_token_slice = torch.tensor(cache_token_id).repeat(input_ids.shape[0], 1).to(input_ids.device)
            # FIXME: Set the min context length as 32 for now, make it configurable.
            insert_idx = torch.randint(low=32, high=model.config.block_size - 32, size=(1,))
            input_ids = torch.cat([input_ids[:, :insert_idx], cache_token_slice, input_ids[:, insert_idx:]], dim=1)
            targets = torch.cat([targets[:, : insert_idx - 1], cache_token_slice, targets[:, insert_idx - 1 :]], dim=1)

        if state["iter_num"] < cfg.shape_watching_iters:
            stdout_log.info(f"bsz: {bsz} | seq_len: {seq_len}")
            stdout_log.info(f"input_ids.shape: {input_ids.shape} | targets.shape: {targets.shape}")
        elif state["iter_num"] == cfg.shape_watching_iters and cfg.shape_watching_iters > 0:
            stdout_log.info("Silencing shape watching ...")

        # Forward, loss, and backward computation.
        is_accumulating = state["iter_num"] % cfg.gradient_accumulation_steps != 0

        # TODO: @Prajwal/@Siddharth Add communication optimiser
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.fabric_strategy == "axonn_tp")):
                logits, loss_mask = model(
                    input_ids,
                    doc_block_attn=cfg.doc_block_attn,
                    cache_attn=cfg.cache_attn,
                    cache_token_id=cache_token_id,
                )
                if cfg.doc_block_attn or cfg.cache_attn:
                    assert loss_mask is not None, "loss_mask must be returned for doc_block_attn or cache_attn"
                    # For cache_attn: Apply the loss mask to the targets.
                    # Loss_mask: 1 for valid target entries and 0 for masked entries.
                    loss_mask = loss_mask.to(targets.dtype)
                    targets *= loss_mask
                    inv_loss_mask = (1 - loss_mask) * -1
                    targets += inv_loss_mask  # Set masked entried as -1 to work with chuncked_cross_entropy.
                    if tokenizer:
                        targets[targets == -1] = tokenizer.pad_id

                if cfg.fabric_strategy == "axonn_tp":
                    targets = targets.to(logits.device)

                # Compute the loss with goldfish if enabled.
                loss = loss_func(logits=logits, targets=targets)

            if cfg.fabric_strategy == "axonn_tp":
                fabric.backward(model, loss / cfg.gradient_accumulation_steps)
            else:
                fabric.backward(loss / cfg.gradient_accumulation_steps)

            with torch.no_grad():
                if cfg.k_goldfish is not None:
                    # No goldfish loss (override k_goldfish to None in `loss_func` for no goldfish loss)
                    all_token_loss = loss_func(logits=logits, targets=targets, goldfish_strategy=None)
                    
                    ignore_index = tokenizer.pad_id if tokenizer else -1
                    goldfish_masked_targets, _ = apply_goldfish(
                                            targets=targets,
                                            strategy=cfg.goldfish_strategy,
                                            k=cfg.k_goldfish,
                                            goldfish_start_position=cfg.goldfish_start_position,
                                            goldfish_context_width=cfg.goldfish_context_width,
                                            ignore_index=ignore_index,
                                        )
                    post_goldfish_token_count = (goldfish_masked_targets != tokenizer.pad_id if tokenizer else -1).sum().item()
                    no_goldfish_token_count = (targets != tokenizer.pad_id if tokenizer else -1).sum().item()
                    total_loss_difference = (all_token_loss * no_goldfish_token_count) - (loss * post_goldfish_token_count)
                    dropped_token_loss = total_loss_difference / (no_goldfish_token_count - post_goldfish_token_count)


        running_loss.update(loss.detach())

        # For goldfish jobs logs only
        if cfg.k_goldfish is not None:
            running_all_token_loss.update(all_token_loss.detach())
            running_dropped_tokens_loss.update(dropped_token_loss.detach())

        # Take an optimization step if not accumulating.
        if not is_accumulating:
            grad_norm = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
            # Update learning rate (post-increment since we init it before the first step).
            lr = (
                get_lr(it=state["iter_num"], lr_decay_iters=cfg.max_iters, cfg=cfg)
                if cfg.decay_lr
                else cfg.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            data_scheduler.step(state["step_count"])

        # Log at an interval.
        if state["iter_num"] % cfg.log_iter_interval == 0:
            log_iter(
                fabric,
                state,
                running_loss,
                running_all_token_loss,
                running_dropped_tokens_loss,
                lr,
                throughput,
                initial_iter,
                total_t0,
                iter_t0,
                grad_norm,
                is_accumulating,
                data_scheduler,
                cfg,
            )

        # Validate at an interval.
        # FIXME Isn't this logic a bit off? If the step interval is divisible by
        # the accumulation steps, could we never hit the save_and_eval_interval?
        if val_dataloader is not None and not is_accumulating and state["step_count"] % cfg.eval_step_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, val_dataloader, max_iters=cfg.eval_iters, cfg=cfg, tokenizer=tokenizer)
            val_loss = val_loss.item()
            td = time.time() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "step": state["step_count"],
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        # Pathing for various save conditions.
        if cfg.fabric_strategy == "axonn_tp":
            fully_qualified_checkpoint_path = f"{cfg.out_dir}/{fabric.get_tensor_parallel_prefix_for_checkpoint()}/step-{state['step_count']:08d}-{cfg.run_name}.pth"
        else:
            fully_qualified_checkpoint_path = f"{cfg.out_dir}/step-{state['step_count']:08d}-{cfg.run_name}.pth"

        # Save at an interval.
        have_not_saved = True
        if not is_accumulating and state["step_count"] % cfg.save_step_interval == 0:
            checkpoint_path = fully_qualified_checkpoint_path
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

            have_not_saved = False

            # TODO, unless prohibitively slow, we should be able to call the
            # scripts.convert_pretrained_checkpoint.convert_checkpoint function here
            # which would turn the training checkpoint into a final saved model.
            # Could even call the lit-to-hf conversion process as well.

        # Save n minutes before job done
        if have_not_saved is True and cfg.save_n_min_before_job_done is not None:
            time_spent = time.time() - global_start_time
            remaining_time = cfg.global_total_time - time_spent
            remaining_time = remaining_time / 60.0
            remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")

            if remaining_time <= cfg.save_n_min_before_job_done:
                checkpoint_path = fully_qualified_checkpoint_path
                fabric.print(
                    f"Saving checkpoint to {str(checkpoint_path)!r}, saving at {remaining_time:.02f} minutes left"
                )
                fabric.save(checkpoint_path, state)

                cfg.save_n_min_before_job_done = None

                have_not_saved = False

    # Last step save and validate.
    if have_not_saved is True and cfg.save_last_step is True:
        t0 = time.time()
        val_loss = validate(fabric, model, val_dataloader, max_iters=cfg.eval_iters, cfg=cfg, tokenizer=tokenizer)
        val_loss = val_loss.item()
        td = time.time() - t0

        fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
        metrics = {
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss),
            "step": state["step_count"],
        }
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.barrier()

        checkpoint_path = fully_qualified_checkpoint_path
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, state)


####################################################################################################
# Train loop sub-routines.
####################################################################################################


def log_iter(
    fabric: L.Fabric = None,
    state: dict = None,
    running_loss: RunningMean = None,
    running_all_token_loss: RunningMean = None,
    running_dropped_tokens_loss: RunningMean = None,
    lr: float = None,
    throughput: ThroughputMonitor = None,
    initial_iter: int = None,
    total_t0: float = None,
    iter_t0: float = None,
    grad_norm: float = None,
    is_accumulating: bool = None,
    data_scheduler: dict = None,
    cfg: CLISettings = None,
):
    """Log the iteration and compute the throughput."""
    loss = running_loss.compute().item()  # expensive device-to-host synchronization # NOTE not sure how true this is.
    running_all_token_loss = running_all_token_loss.compute().item() if running_all_token_loss is not None else None
    running_dropped_tokens_loss = running_dropped_tokens_loss.compute().item() \
        if running_dropped_tokens_loss is not None else None
    t1 = time.time()

    # Log the metrics.
    # FIXME, token counting logic assumes fixed microbatch size w/ no padding.
    # This is fine for pretraining style data, but this might not always be true.
    metrics = {
        "loss": loss,
        "loss_all_tokens": running_all_token_loss,
        "loss_dropped_tokens": running_dropped_tokens_loss,
        "iter": state["iter_num"],
        "step": state["step_count"],
        "iter_time": t1 - iter_t0,
        "remaining_time": ((t1 - total_t0) / (state["iter_num"] - initial_iter) * (cfg.max_iters - state["iter_num"])),
        "tokens": state["iter_num"] * cfg.micro_batch_size * cfg.block_size,
        "total_tokens": state["iter_num"] * cfg.micro_batch_size * cfg.block_size * fabric.world_size,
        "learning_rate": lr,
        "max_iters": cfg.max_iters,
        "grad_norm": grad_norm,
    }

    if cfg.fabric_strategy != "axonn_tp":
        throughput.update(
            time=(t1 - total_t0),
            flops=(cfg.measured_flops * cfg.log_iter_interval),
            batches=state["iter_num"],
            samples=(state["iter_num"] * cfg.micro_batch_size),
            lengths=(state["iter_num"] * cfg.micro_batch_size * cfg.block_size),
        )
        throughput_metrics = throughput.compute()
        metrics.update(throughput_metrics)

    # Update loss and grad_norm with all_reduce
    # FIXME _these_ could be expensive if the topo is large, so do we need to always report
    # world reduced loss or is rank-local loss sufficient? Maybe add a flag option.
    loss = fabric.all_reduce(loss)
    loss_all_tokens = fabric.all_reduce(running_all_token_loss) if cfg.goldfish_strategy is not None else None
    loss_dropped_tokens = fabric.all_reduce(running_dropped_tokens_loss) if cfg.goldfish_strategy is not None else None
    grad_norm = fabric.all_reduce(grad_norm)
    metrics["loss"] = loss
    metrics["grad_norm"] = grad_norm
    metrics["loss_all_tokens"] = loss_all_tokens
    metrics["loss_dropped_tokens"] = loss_dropped_tokens

    if data_scheduler is not None:
        curr_data_weights = data_scheduler.get_data_weights()
        curr_data_weights = dict(zip(cfg.dataset_names, curr_data_weights))

        curr_sample_count = data_scheduler.get_sample_count()
        curr_sample_count = fabric.all_reduce(curr_sample_count, reduce_op="sum")

        curr_epoch_count = data_scheduler.get_epoch_count()
        curr_epoch_count = fabric.all_reduce(curr_epoch_count, reduce_op="mean")

        for i, x in enumerate(curr_data_weights.keys()):
            metrics["data_scheduler_weight/" + x] = curr_data_weights[x]
            metrics["data_scheduler_norm_weight/" + x] = curr_data_weights[x] / sum(list(curr_data_weights.values()))

            metrics["data_scheduler_sample_count/" + x] = curr_sample_count[i]
            metrics["data_scheduler_epoch_count/" + x] = curr_epoch_count[i]

    fabric.log_dict(metrics, step=state["iter_num"])

    # Log some metrics to the console.

    # goldfish Losses if goldfish is enabled.
    goldfish_specific_loss = (
        f"loss_all_tokens: {metrics['loss_all_tokens']:.4f}, " \
        f"loss_dropped_tokens: {metrics['loss_dropped_tokens']:.4f}, " if cfg.goldfish_strategy is not None else ""
    )

    fabric.print(
        f" iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, {goldfish_specific_loss} iter time:"
        f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step),' if not is_accumulating else ','}"
        f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
        f" max iters: {metrics['max_iters']}"
        f" grad norm: {metrics['grad_norm']:.4f}"
        f" learning rate: {metrics['learning_rate']}"
        f" total tokens: {metrics['total_tokens']}"
    )
    pass


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    max_iters: int,
    cfg: CLISettings,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    fabric.print("Validating ...")
    fabric.print(f"Max iters: {max_iters}")

    model.eval()

    losses = torch.zeros(max_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= max_iters:
            break
        bsz, seq_len = val_data.shape
        input_ids = val_data[:, 0 : (seq_len - 1)].contiguous().long()
        if cfg.fabric_strategy == "axonn_tp":
            input_ids = input_ids.cuda()
        # for the input we need to replace any pad ids with the eos token
        # knowing that they're trailing, and wont contrib to activations/loss
        # but that they need to be valid indices in the model's embedding layer
        if tokenizer:
            input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id

        targets = val_data[:, 1:(seq_len)].contiguous().long()

        # TODO: @Prajwal/@Siddharth Add communication optimiser
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.fabric_strategy == "axonn_tp")):
            # TODO: need another val function with cache token enabled for caching ability.
            logits, loss_mask = model(input_ids, doc_block_attn=cfg.doc_block_attn)
            if cfg.doc_block_attn or cfg.cache_attn:
                assert loss_mask is not None, "loss_mask must be returned for doc_block_attn or cache_attn"
                # For cache_attn: Apply the loss mask to the targets.
                # Loss_mask: 1 for valid target entries and 0 for masked entries.
                loss_mask = loss_mask.to(targets.dtype)
                targets *= loss_mask
                inv_loss_mask = (1 - loss_mask) * -1
                targets += inv_loss_mask  # Set masked entried as -1 to work with chuncked_cross_entropy.
                if tokenizer:
                    targets[targets == -1] = tokenizer.pad_id
                # No goldfish when reporting validation loss.

            if cfg.fabric_strategy == "axonn_tp":
                targets = targets.to(logits.device)

            if tokenizer:
                loss = chunked_cross_entropy(logits, targets.to(logits.device), ignore_index=tokenizer.pad_id)
            else:
                loss = chunked_cross_entropy(logits, targets.to(logits.device))
        losses[k] = loss

    model.train()
    return losses.mean()


####################################################################################################
# Data utility functions.
####################################################################################################


def create_dataloader(
    data_config: dict,
    batch_size: int,
    block_size: int,
    n_chunks: int,
    data_dir: Path,
    fabric: L.Fabric,
    shuffle: bool = True,
    seed: int = 1337,
    cfg: CLISettings = None,
    tokenizer: Tokenizer = None,
) -> DataLoader:
    global_data_dir = data_dir
    datasets = []
    for curr_config in data_config:

        if curr_config["type"] == "hfds":
            assert tokenizer is not None, "tokenizer must be provided for HuggingfaceDataset"
            assert "data_dir" in curr_config, "data_dir must be provided for HuggingfaceDataset"
            dataset = HuggingfaceDataset(
                ds_name_or_path=curr_config["data_dir"],  # this is a path to a previously save_to_disk'd hfds
                seed=seed,
                shuffle=shuffle,
                num_processes=(
                    fabric.global_world_size_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.world_size
                ),
                process_rank=(
                    fabric.global_rank_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.global_rank
                ),
                shortname=curr_config["prefix"],  # this is provided for logging, and schedule purposes
                text_key=curr_config.get("text_key", cfg.text_key),  # key for the field in dataset to return
                repetitions=curr_config.get("repetitions"),  # repeat the dataset a number of times
            )

        elif curr_config["type"] == "pkds":
            prefix = curr_config["prefix"]

            if "data_dir" in curr_config:
                data_dir = curr_config["data_dir"]
            else:
                data_dir = global_data_dir

            if fabric.global_rank == 0:
                filenames = [str(pth) for pth in sorted(Path(data_dir).glob(f"{prefix}*"))]
                if cfg.shuffle_filenames:
                    random.seed(seed)
                    random.shuffle(filenames)  # inplace
                if not filenames:
                    raise FileNotFoundError(f"No files found at {str(data_dir)} with prefix {prefix}.")
            else:
                filenames = None

            filenames = fabric.broadcast(filenames, 0)  # this is a blocking op from rank 0 to all other ranks

            # log after broadcast so we know we passed it.
            if fabric.global_rank == 0:
                num_processes = (
                    fabric.global_world_size_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.world_size
                )
                process_rank = (
                    fabric.global_rank_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.global_rank
                )
                stdout_log.info(
                    f"Rank ({process_rank}/{num_processes}) glob'd {len(filenames)} files"
                    f" from {data_dir}{f' w/ prefix {prefix}' if prefix not in ['','*'] else ''},"
                    f" files[:3]: {filenames[:3]}"
                )

            dataset = PackedDataset(
                filenames,
                n_chunks=n_chunks,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed,
                num_processes=(
                    fabric.global_world_size_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.world_size
                ),
                process_rank=(
                    fabric.global_rank_for_creating_dataloader
                    if cfg.fabric_strategy == "axonn_tp"
                    else fabric.global_rank
                ),
                shortname=prefix,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {curr_config['type']}")

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [curr_config["weight"] for curr_config in data_config]
    data_scheduler_tracker = DataSchedulerTracker(weights)

    combined_dataset = CombinedDataset(
        datasets=datasets, seed=seed, data_scheduler_tracker=data_scheduler_tracker, data_telemetry=cfg.data_telemetry
    )

    parametrized_collate_fn = partial(
        generic_collate_fn,
        tokenizer=tokenizer,
        block_size=cfg.loader_block_size,
        pad_to_block_size=cfg.pad_to_block_size,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        collate_checks_enabled=cfg.collate_checks_enabled,
        all_block_size_tensors=cfg.all_block_size_tensors,
    )

    return (
        DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=parametrized_collate_fn,
            num_workers=cfg.dataloader_num_workers,
        ),
        data_scheduler_tracker,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    seed: int = 1337,
    cfg: CLISettings = None,
    tokenizer: Tokenizer = None,
) -> Tuple[DataLoader, DataLoader]:

    stdout_log.info(f"Creating dataloaders with seed: {seed}")
    train_dataloader, data_scheduler_tracker = create_dataloader(
        cfg.data_config["train_data"],
        batch_size=batch_size,
        block_size=block_size,
        n_chunks=cfg.n_chunks,
        fabric=fabric,
        data_dir=cfg.train_data_dir,
        shuffle=True,
        seed=seed,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    val_dataloader, _ = (
        create_dataloader(
            cfg.data_config["val_data"],
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            shuffle=False,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
        )
        if cfg.val_data_dir
        else None
    )
    return train_dataloader, val_dataloader, data_scheduler_tracker


####################################################################################################
# Train utility functions.
####################################################################################################


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int, lr_decay_iters: int, cfg: CLISettings) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    if cfg.lr_schedule == "linear":
        return cfg.learning_rate - decay_ratio * (cfg.learning_rate - cfg.min_lr)
    elif cfg.lr_schedule == "constant":
        return cfg.learning_rate
    elif cfg.lr_schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def init_weights(module: nn.Module, n_layer: int, n_embd: int, axonn_tp: bool = False):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if not axonn_tp:
        # AxoNN does the inits internally for its linear layers, so we will skip this
        for name, param in module.named_parameters():
            if name == "proj.weight" and isinstance(module, (LLaMAMLP, CausalSelfAttention)):
                nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(n_embd) / n_layer))


if __name__ == "__main__":
    cfg = CLI(CLISettings)

    # Next we set up the fabric and logger.
    setup_fabric(cfg)

    # Then these functions are called in order using tail calls:
    # [cfg from CLI] -> setup_fabric -> main -> train

    # FWIW this chain call design is different than I'd normally do it, but it's what we have for now.
