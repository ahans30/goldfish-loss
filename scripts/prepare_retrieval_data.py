# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import json
import os
import sys
from pathlib import Path
from typing import Union
from functools import partial

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer
import torch


def prepare(
    destination_path: Path = Path("/fs/cml-projects/llm-pretraining/llm-retrieval/data/orca_retrieval"),
    checkpoint_dir: Path = Path("/fs/cml-projects/llm-pretraining/llm-retrieval/checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    seed: int = 42,
    cache_dir: Path = Path("/fs/cml-projects/llm-pretraining/llm-retrieval/data/cache"),
    test_size: Union[float, int, None] = 0.0005,
    max_seq_length: int = None,
    data_name: str = "openwebtext",
    data_type: str = "pretrain"
) -> None:
    np.random.seed(seed)
    from datasets import load_dataset  # huggingface datasets

    destination_path.mkdir(parents=True, exist_ok=True)

    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    tokenizer = Tokenizer(checkpoint_dir)

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = os.cpu_count() // 2

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on HW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset(data_name, num_proc=num_proc_load_dataset, cache_dir=cache_dir)
    test_size = 10000 / len(dataset['train']) # picking 10000 samples for test set
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    val_dataset = split_dataset.pop("test")  # rename the test split to val

    def process_instruction_data(examples, max_length=1024):
        # writing for batched examples
        query_ids = []
        corpus_ids = []
        lens = []
        for question, response in zip(examples["question"], examples["response"]):
            query_id = tokenizer.encode(question, max_length=max_length, bos=False, eos=False).tolist()   # not adding bos, eos for now
            corpus_id = tokenizer.encode(response, max_length=max_length).tolist()   # not adding bos, eos for now
            if len(query_id) <= max_length and len(corpus_id) <= max_length:
                query_ids.append(query_id)
                corpus_ids.append(corpus_id)

        return {"query": query_ids, "corpus": corpus_ids, 'query_len': [len(q) for q in query_ids], 'corpus_len': [len(c) for c in corpus_ids]}

    def process_pretrain_data(examples, max_length=1024):
        # writing for batched examples
        query_ids = []
        corpus_ids = []
        lens = []
        for text in examples["text"]:
            # splitting the text at random points and make query and corpus
            tokenized_text = tokenizer.encode(text, max_length=max_length, bos=False, eos=False).tolist()
            if len(tokenized_text) > 8: # making a random choice that the query and corpus are not too small
                pos = np.random.randint(5, len(tokenized_text))
                query_id = tokenized_text[:pos]
                corpus_id = tokenized_text[pos:]
                query_ids.append(query_id)
                corpus_ids.append(corpus_id)

        return {"query": query_ids, "corpus": corpus_ids, 'query_len': [len(q) for q in query_ids], 'corpus_len': [len(c) for c in corpus_ids]}

    # tokenize the dataset
    if data_type == "pretrain":
        tokenize_func = partial(process_pretrain_data, max_length=max_seq_length)
    elif data_type == "instruction":
        tokenize_func = partial(process_instruction_data, max_length=max_seq_length)
    else:
        raise ValueError(f"Invalid data_type: {data_type}; Please choose from 'pretrain' or 'instruction'")
    tokenized = val_dataset.map(tokenize_func, desc="tokenizing the splits", batched=True, num_proc=num_proc)
    # removing all columns except query and corpus
    tokenized = tokenized.remove_columns([col for col in tokenized.column_names if col not in ["query", "corpus", "query_len", "corpus_len"]])
    tokenized = tokenized.add_column("qrel", range(len(tokenized)))
    # saving as hf dataset
    tokenized.save_to_disk(destination_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
