# Written based on prepare_redpajma.py and prepare_slimpajama.py from lit_gpt repo.
import hashlib
import glob
import json
import os
import sys
from pathlib import Path
import logging
from time import time
from multiprocessing import cpu_count, Pool

from tqdm import tqdm
import numpy as np
from torch import tensor

from functools import partial

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from datasets.utils.py_utils import convert_file_size_to_int
from datasets import load_dataset, DatasetDict, load_from_disk

import lit_gpt.packed_cycle_dataset as packed_dataset
from lit_gpt import Config, Tokenizer

logging.basicConfig(level=logging.INFO)


# these are demos,integrating a bit more is still TODO
def prepare_text(row, text_field="text"):
    """Passthrough for text fields (or other single, named fields)."""
    return row.get(text_field)


def prepare_openorca(row, **kwargs):
    """Example using the OpenOrca fields."""
    dbl_nl = "\n\n"
    return (
        f"{row['system_prompt']+dbl_nl if row['system_prompt']!='' else ''}{row['question']}{dbl_nl}{row['response']}"
    )


def prepare_ultrachat(row, **kwargs):
    """Example for UltraChat format."""
    sgl_nl = "\n"
    dbl_nl = "\n\n"
    dialog_turns = [f"{x['role'].capitalize()}:{sgl_nl}{x['content']}" for x in row["messages"]]
    return dialog_turns


def prepare_openorca_chatml(row, **kwargs):
    if row["response"].isspace() or row["response"] == "":
        return ""

    dialog = f"<|im_start|>user\n{row['question']}\n<|im_end|>\n<|im_start|>assistant\n{row['response']}\n<|im_end|>"

    return dialog


def prepare_ultrachat_chatml(row, **kwargs):
    dialog_turns = ""
    for i in range(len(row["data"])):
        curr_data = row["data"][i]

        if curr_data.isspace() or curr_data == "":
            return ""

        if i % 2 == 0:
            dialog_turns += f"<|im_start|>user\n{curr_data}\n<|im_end|>"
        else:
            dialog_turns += f"<|im_start|>assistant\n{curr_data}\n<|im_end|>"

        if i != len(row["data"]) - 1:
            dialog_turns += "\n"

    return dialog_turns


def prepare_openorca_assistant(row, **kwargs):
    if row["response"].isspace() or row["response"] == "":
        return ""

    dialog = f"<|user|>\n{row['question']}\n<|assistant|>\n{row['response']}"

    return dialog


def prepare_ultrachat_assistant(row, **kwargs):
    dialog_turns = ""
    for i in range(len(row["data"])):
        curr_data = row["data"][i]

        if curr_data.isspace() or curr_data == "":
            return ""

        if i % 2 == 0:
            dialog_turns += f"<|user|>\n{curr_data}"
        else:
            dialog_turns += f"<|assistant|>\n{curr_data}"

        if i != len(row["data"]) - 1:
            dialog_turns += "\n"

    return dialog_turns


PREPARE_FN_MAP = {
    "default": prepare_text,
    "openorca": prepare_openorca,
    "ultrachat": prepare_ultrachat,
    "openorca_chatml": prepare_openorca_chatml,
    "ultrachat_chatml": prepare_ultrachat_chatml,
    "openorca_assistant": prepare_openorca_assistant,
    "ultrachat_assistant": prepare_ultrachat_assistant,
}


def prepare_metadata_openorca(row, **kwargs):
    """Example using the OpenOrca fields."""
    return f"{row['id'].split('.')[0]}"


def prepare_metadata_flan(row, **kwargs):
    """Example using the FLAN fields."""
    return f"{row['_task_name']}"


METADATA_FN_MAP = {
    "openorca": prepare_metadata_openorca,
    "flan": prepare_metadata_flan,
}


def shard_name(prefix, shard_index, num_shards):
    return f"{prefix}_{shard_index:06d}-of-{num_shards:06d}"


def build_shard(
    shard_index,
    dataset,
    text_column,
    ds_shortname,
    num_shards,
    destination_path,
    prefix,
    chunk_size,
    tokenizer,
    add_bos=None,
    add_eos=None,
    skip_remainder=None,
    randomize_tokens=None,
):
    """Build a shard by writing to a PackedDataset object. This function defines the work of one shard."""
    shard = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    # Note that this shard contains all columns from the original dataset, not just the text column.
    # we can use these if we'd like to.

    # by default we pad the tail of each array with EOS.
    sep_token = tokenizer.eos_id
    # only if we're _only_ adding BOS tokens do we use that instead.
    if add_bos and not add_eos:
        sep_token = tokenizer.bos_id

    assert sep_token is not None, "Tokenizer does not have the expected token for use as separator."

    if randomize_tokens:
        # we want to draw random tokens to replace all tokens are not BOS, EOS, or PAD,
        # but we don't want to draw any of those again either
        bos_id, eos_id, pad_id = tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id
        unk_id = 0  # unk is hardcoded for now, not sure how to automatically get it
        logging.info(f"Randomizing tokens in shard {shard_index} of {num_shards}")
        logging.warning(
            f"The tokens to be avoided during random sampling are: bos={bos_id}, eos={eos_id}, pad={pad_id} and hardcoded 'unk'={unk_id}. All other tokens may be sampled as replacements."
        )
        valid_vocab_indices_range = np.arange(tokenizer.vocab_size)
        valid_vocab_indices = np.setdiff1d(valid_vocab_indices_range, [bos_id, eos_id, pad_id])

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=shard_name(prefix, shard_index, num_shards),
        chunk_size=chunk_size,
        sep_token=sep_token,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    def process_row(row, bldr=None, tokzr=None, add_eos=None, add_bos=None):
        # NOTE: we could make this batched without messing things up I think.

        texts = PREPARE_FN_MAP[ds_shortname](row, text_field=text_column)

        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            if text == "":
                return
            elif text is None:
                raise ValueError(
                    f"Row {row} does not contain expected columns (maybe 'text' is named something else?)."
                )
            else:
                text_ids = tokzr.encode(text, bos=add_bos, eos=add_eos)

                if randomize_tokens:
                    valid_replace_indices = np.where(
                        np.logical_and(np.logical_and(text_ids != bos_id, text_ids != eos_id), text_ids != pad_id)
                    )[0]
                    random_replacements = np.random.choice(valid_vocab_indices, valid_replace_indices.shape[0])
                    text_ids[valid_replace_indices] = tensor(random_replacements, dtype=text_ids.dtype)

                bldr.add_array(np.array(text_ids, dtype=bldr.dtype))

    process_partial = partial(process_row, bldr=builder, tokzr=tokenizer, add_eos=add_eos, add_bos=add_bos)

    shard.map(
        process_partial,
        num_proc=1,  # no nested parallelism
        desc=f"Processing {prefix} shard {shard_index:06d} of {num_shards:06d}",
    )

    # If flag is set, we skip the remainder of the last chunk avoiding writing a bunch of filler,
    # but sacrificing some valid tokens.
    if skip_remainder:
        skipped_tokens = builder.skip_write_remainder()
    else:
        skipped_tokens = 0
        builder.write_remainder()

    # token counting.
    # We can get an estimate for the builder by the chunk size * number of chunks
    # though it's technically an upper bound. We could tighten this further by
    # comparing the last chunk's size on disk to the rest of the chunks.

    chunks_written = builder._counter
    # We can use internal counters to get the exact number of tokens written
    tokens_written = builder._total_tokens_exact
    filler_sep_tokens_written = builder._filler_sep_tokens
    # logging.info(f"Shard {shard_index:06d} of {num_shards:06d} contained {tokens_written} tokens")
    logging.info(
        f"Shard {shard_index:06d} of {num_shards:06d} contained {tokens_written} tokens (and {filler_sep_tokens_written} filler sep tokens, skipped {skipped_tokens} tokens in remainder chunk)"
    )

    return (
        tokens_written,
        filler_sep_tokens_written,
        skipped_tokens,
        chunks_written,
    )


def format_tokens(total_tokens):
    if total_tokens > 1_000_000_000:
        token_ct_str = f"{total_tokens/1_000_000_000:.1f}B"
    elif total_tokens > 1_000_000:
        token_ct_str = f"{total_tokens/1_000_000:.1f}M"
    elif total_tokens > 1_000:
        token_ct_str = f"{total_tokens/1_000:.1f}K"
    else:
        token_ct_str = f"{total_tokens}"
    return token_ct_str


def build_dataset(
    dataset,
    text_column,
    ds_shortname,
    tokenizer,
    destination_path,
    prefix,
    chunk_size,
    num_shards,
    shard_size,
    num_proc,
    add_bos,
    add_eos,
    skip_remainder,
    randomize_tokens,
):
    """Build a dataset by writing to a PackedDataset.
    This funtion defines the dataset building and work partitioning logic and launches the worker pool.
    """

    if num_shards is None:
        shard_size = shard_size

        max_shard_size = convert_file_size_to_int(shard_size)
        dataset_nbytes = convert_file_size_to_int(dataset.data.nbytes)
        num_shards = int(dataset_nbytes / max_shard_size) + 1
        num_shards = max(num_shards, 1)

        logging.info(f"Auto-derived sharding parameters:")
        logging.info(f"Dataset len = {len(dataset)/1_000_000:.2f} M rows")
        logging.info(f"Dataset full size = {dataset_nbytes/1_000_000_000:.2f} GB")
        logging.info(f"Target shard size = {shard_size}")
        logging.info(f"Number of shards = {num_shards}")
    else:
        assert num_shards > 0
        logging.info(f"Using user-defined sharding parameters:")
        logging.info(f"Number of shards = {num_shards}")

    if num_shards > len(dataset):
        logging.warning(f"Number of shards ({num_shards}) is greater than the dataset length ({len(dataset)}).")
        num_shards = len(dataset)
        logging.warning(f"Setting number of shards to {num_shards}.")

    shard_indices = list(range(num_shards))
    shard_partial = partial(
        build_shard,
        dataset=dataset,
        text_column=text_column,
        ds_shortname=ds_shortname,
        num_shards=num_shards,
        destination_path=destination_path,
        prefix=prefix,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
        add_bos=add_bos,
        add_eos=add_eos,
        skip_remainder=skip_remainder,
        randomize_tokens=randomize_tokens,
    )

    pool_size = min(num_shards, num_proc)

    logging.info(f"Building dataset w/ {num_shards} shards using {pool_size} processes...")

    start_time = time()

    if pool_size == 1:
        # single process
        results = [shard_partial(shard_index) for shard_index in shard_indices]
    else:
        # multiprocess
        process_pool = Pool(pool_size)

        results = process_pool.map(shard_partial, shard_indices)

        process_pool.close()
        process_pool.join()

    logging.info(f"Building finished! Took {(time()-start_time)/60:.1f}mins")

    logging.info(f"Dataset written to {destination_path}")

    # we can report the total number of tokens written by summing the results
    valid_token_counts = []
    sep_token_counts = []
    skipped_token_counts = []
    valid_chunk_counts = []
    for i, res in enumerate(results):
        if isinstance(res, tuple):
            valid_token_counts.append(res[0])
            sep_token_counts.append(res[1])
            skipped_token_counts.append(res[2])
            valid_chunk_counts.append(res[3])
        else:
            logging.info(f"Shard {i} may have failed to build, worker returned: {res}")

    total_tokens = sum(valid_token_counts)
    total_sep_tokens = sum(sep_token_counts)
    total_skipped_tokens = sum(skipped_token_counts)
    total_chunks = sum(valid_chunk_counts)
    logging.info(f"Total chunks/files written across all {num_shards} shards: {total_chunks}")

    token_ct_str = format_tokens(total_tokens)
    sep_token_ct_str = format_tokens(total_sep_tokens)
    skip_token_ct_str = format_tokens(total_skipped_tokens)

    logging.info(f"Total tokens written across all chunks in all shards: {token_ct_str}")
    logging.info(f"Total separator tokens written across all chunks in all shards: {sep_token_ct_str}")
    logging.info(f"Total skipped tokens across all shards: {skip_token_ct_str}")
    logging.info(f"Packing overhead ratio: {sep_token_ct_str} / {token_ct_str} = {total_sep_tokens/total_tokens:.1%}")

    return num_shards, total_chunks, total_tokens, total_sep_tokens, total_skipped_tokens


def partition_by_meta_column(dataset, prefix_value, num_proc, reduce_to_hashnames=False):

    meta_col_name = "metadata"
    fn = METADATA_FN_MAP[prefix_value]
    ds_w_metadata_column = dataset.map(
        lambda x: {meta_col_name: fn(x)},
        num_proc=num_proc,
        desc=f"Adding {meta_col_name} column for prefix partitioning",
    )

    def hashname_fn(s, hashname_table):
        h = hashlib.shake_128(s.encode()).hexdigest(4)
        hashname_table[s] = h
        return h

    def get_hashname_table(unique_meta_values):
        hashname_table = {}
        num_unique_meta_values = len(unique_meta_values)
        unique_meta_values = [hashname_fn(x, hashname_table) for x in unique_meta_values]
        assert len(set(hashname_table.values())) == num_unique_meta_values, "Hashname collision!"

        return hashname_table, unique_meta_values

    global_hashname_table = {}
    new_ds_dict = DatasetDict()
    if isinstance(dataset, DatasetDict):
        # this means we have a split dataset, already a DatasetDict
        # and we need to do this for each split postpending the prefix to the split name
        for split in dataset.keys():
            unique_meta_values = ds_w_metadata_column[split].unique(meta_col_name)
            if reduce_to_hashnames:
                hashname_table, unique_meta_values = get_hashname_table(unique_meta_values)
                global_hashname_table.update(hashname_table)

            for meta_value in unique_meta_values:
                if reduce_to_hashnames:
                    meta_compare_fn = lambda x: hashname_table[x[meta_col_name]] == meta_value
                else:
                    meta_compare_fn = lambda x: x[meta_col_name] == meta_value

                new_ds_dict[f"{split}-meta-{meta_value}"] = ds_w_metadata_column[split].filter(
                    meta_compare_fn,
                    num_proc=num_proc,
                    desc=f"Filtering {split} for {meta_col_name}=={meta_value}",
                )
    else:
        # this means we have a single dataset, not a DatasetDict
        unique_meta_values = ds_w_metadata_column.unique(meta_col_name)
        if reduce_to_hashnames:
            hashname_table, unique_meta_values = get_hashname_table(unique_meta_values)
            global_hashname_table.update(hashname_table)

        for meta_value in unique_meta_values:
            if reduce_to_hashnames:
                meta_compare_fn = lambda x: hashname_table[x[meta_col_name]] == meta_value
            else:
                meta_compare_fn = lambda x: x[meta_col_name] == meta_value

            new_ds_dict[f"meta-{meta_value}"] = ds_w_metadata_column.filter(
                meta_compare_fn,
                num_proc=num_proc,
                desc=f"Filtering for {meta_col_name}=={meta_value}",
            )

    return new_ds_dict, global_hashname_table


def prepare_packed_dataset(
    dataset_name_or_path: str = None,
    dataset_config: str = None,
    dataset_kwargs: str = None,
    text_column: str = None,
    ds_shortname: str = None,
    prefix_type: str = None,
    prefix_value: str = None,
    reduce_to_hashnames: bool = None,
    checkpoint_dir: Path = None,
    destination_path: Path = None,
    chunk_size: int = None,
    num_proc: int = None,
    num_shards: int = None,
    shard_size: str = None,
    ld_from_disk: bool = None,
    add_bos: bool = None,
    add_eos: bool = None,
    cleanup_cache_files: bool = None,
    skip_remainder: bool = None,
    randomize_tokens: bool = None,
) -> None:
    """Prepare the dataset by writing to a PackedDataset. This funtion defines the HF Dataset loading logic."""

    # handle ds kwargs
    if dataset_kwargs is not None:
        parsed_kwargs = {}
        for kwarg in dataset_kwargs.split(","):
            k, v = kwarg.split("=")
            parsed_kwargs[k] = v
        dataset_kwargs = parsed_kwargs
    else:
        dataset_kwargs = {}

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    logging.info(
        f"Downloading {dataset_name_or_path} to {destination_path} with {num_proc} processes w/ {dataset_kwargs} extra args ..."
    )
    start_time = time()
    if ld_from_disk:
        dataset = load_from_disk(dataset_name_or_path)
    else:
        if dataset_config is not None:
            dataset = load_dataset(
                dataset_name_or_path,
                dataset_config,
                **dataset_kwargs,
                num_proc=num_proc,
            )
        else:
            dataset = load_dataset(
                dataset_name_or_path,
                **dataset_kwargs,
                num_proc=num_proc,
            )
    logging.info(f"load_dataset took {(time()-start_time)/3600}hrs")

    logging.info(dataset)

    if isinstance(dataset, DatasetDict):
        splits = dataset.keys()
    else:
        assert prefix_type != "split"
        splits = None

    if prefix_type == "split":
        prefixes = splits
    elif prefix_type == "dataset_config":
        prefixes = [dataset_config]
    elif prefix_type == "manual":
        prefixes = [prefix_value]
    elif prefix_type == "meta_column":
        prefixes = []
    elif prefix_type is None:
        prefixes = ["data"]
    else:
        raise ValueError(f"Unknown prefix_type: {prefix_type}")

    # For the meta_column prefix type, we're build a dataset dict with the unique values of the meta_column
    # as the keys, and set of filtered datasets as the values.
    # The meta_column value is passed as the prefix to the build_dataset function.
    if prefix_type == "meta_column":
        dataset, hashname_table = partition_by_meta_column(dataset, prefix_value, num_proc, reduce_to_hashnames)
        prefixes = list(dataset.keys())
        # we'll save the hashname table to the destination path
        if hashname_table != {}:
            with open(destination_path / "metadata_hashname_table.json", "w") as f:
                json.dump(hashname_table, f, indent=4)

    all_shards, all_chunks, all_tokens, all_sep_tokens, all_skipped_tokens = [], [], [], [], []

    # Warn about autosharding when we have multiple prefixes
    if num_shards is None and len(prefixes) > 1:
        logging.info(
            f"NOTE: auto sharding based on size uses the size on disk of the full underlying dataset. Will overshard smaller prefixes."
        )

    for prefix in prefixes:
        if isinstance(dataset, DatasetDict):
            ds = dataset[prefix]
        else:
            ds = dataset

        prefix_num_shards, prefix_num_chunks, prefix_num_tokens, prefix_sep_tokens, prefix_skipped_tokens = (
            build_dataset(
                dataset=ds,
                text_column=text_column,
                ds_shortname=ds_shortname,
                tokenizer=tokenizer,
                destination_path=destination_path,
                prefix=prefix,
                chunk_size=chunk_size,
                num_shards=num_shards,
                shard_size=shard_size,
                num_proc=num_proc,
                add_bos=add_bos,
                add_eos=add_eos,
                skip_remainder=skip_remainder,
                randomize_tokens=randomize_tokens,
            )
        )
        all_shards.append(prefix_num_shards)
        all_chunks.append(prefix_num_chunks)
        all_tokens.append(prefix_num_tokens)
        all_sep_tokens.append(prefix_sep_tokens)
        all_skipped_tokens.append(prefix_skipped_tokens)

    if cleanup_cache_files:
        res = dataset.cleanup_cache_files()
        logging.info(f"Cleanup cache files returned: {res}")

    if len(prefixes) > 1:
        logging.info(f"Per-prefix stats:")
        for i, prefix in enumerate(prefixes):
            logging.info(
                f"Prefix {prefix} contained {all_shards[i]} shards with {all_chunks[i]} chunks containing {format_tokens(all_tokens[i])} tokens."
            )
        logging.info(f"Summary over all data:")
        logging.info(f"Total chunks/files written across all {sum(all_shards)} work shards: {sum(all_chunks)}")
        logging.info(f"Total tokens written for entire dataset: {format_tokens(sum(all_tokens))}")
        logging.info(f"Total separator tokens written for entire dataset: {format_tokens(sum(all_sep_tokens))}")
        logging.info(f"Total skipped tokens for entire dataset: {format_tokens(sum(all_skipped_tokens))}")
        logging.info(
            f"Total packing overhead ratio: {format_tokens(sum(all_sep_tokens))} / {format_tokens(sum(all_tokens))} = {sum(all_sep_tokens)/sum(all_tokens):.1%}"
        )


def prepare(
    dataset_name_or_path: str = "Jackmin108/c4-en-validation-mini",
    dataset_config: str = None,
    dataset_kwargs: str = None,
    text_column: str = "text",
    ds_shortname: str = "default",  # "default" or a name to access a fn in PREPARE_FN_MAP
    prefix_type: str = "split",  # "split", "dataset_config", "manual"
    prefix_value: str = None,
    reduce_to_hashnames: bool = False,
    checkpoint_dir: Path = Path(
        "/lustre/orion/csc569/scratch/jkirchen/llm-pretraining-root/input/models/meta-llama/Llama-2-7b-chat-hf"
    ),
    destination_path: Path = Path("prepared_hf_dataset"),
    num_proc: int = cpu_count(),
    num_shards: int = None,
    shard_size: str = "500MB",
    chunk_size: int = (2048 + 1) * 16,  # block size + 1 for causal, 16 blocks
    ld_from_disk: bool = False,  # activate this flag if the hf dataset is stored in disk w/ `save_to_disk()` method
    add_bos: bool = False,
    add_eos: bool = True,
    cleanup_cache_files: bool = True,
    skip_remainder: bool = False,
    randomize_tokens: bool = False,
) -> None:
    """Prepare the requested dataset.
    We assume a (hf) tokenizer has been trained and is accessible at the provided path.
    This funtion defines the CLI."""

    logging.info(f"Running with num_proc={num_proc} on a machine with {cpu_count()} visible cpus.")

    prepare_packed_dataset(
        dataset_name_or_path=dataset_name_or_path,
        dataset_config=dataset_config,
        dataset_kwargs=dataset_kwargs,
        text_column=text_column,
        ds_shortname=ds_shortname,
        prefix_type=prefix_type,
        prefix_value=prefix_value,
        reduce_to_hashnames=reduce_to_hashnames,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=chunk_size,
        num_proc=num_proc,
        num_shards=num_shards,
        shard_size=shard_size,
        ld_from_disk=ld_from_disk,
        add_bos=add_bos,
        add_eos=add_eos,
        cleanup_cache_files=cleanup_cache_files,
        skip_remainder=skip_remainder,
        randomize_tokens=randomize_tokens,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
