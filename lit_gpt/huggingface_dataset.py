# Build based on the original code from Lightning AI
# lit_gpt/packed_dataset.py

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct
import hashlib

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# We will build v0 assuming that the dataset is already saved to disk
# in standard hf format. This leaves room for preproc ops as separate logic.
# basic assumpution will be "text" field only.
from datasets import load_from_disk, DatasetDict, Dataset, concatenate_datasets

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


class HuggingfaceDataset(IterableDataset):
    def __init__(
        self,
        ds_name_or_path=None,
        seed=12345,
        shuffle=True,
        num_processes=1,
        process_rank=0,
        shortname=None,
        text_key="text",
        repetitions=None,
    ):
        assert ds_name_or_path is not None
        self._ds_name_or_path = ds_name_or_path
        self._seed = seed
        self._shuffle = shuffle
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._shortname = shortname  # This is human readble, the mixture unit
        self._ds_fingerprint = (
            None  # This is not human readable, corresp to the subset of work _this_ process is handling.
        )
        self._text_key = text_key
        self._ds_total_length = None
        self._ds_length = None
        self._ds = None
        self._subds = None
        self._ds_min = None
        self._ds_max = None

        # Here is where we load the dataset from disk (whole thing, but just the memmap ofc)
        if repetitions is not None:
            ds_list = [load_from_disk(ds_name_or_path) for _ in range(repetitions)]
            self._ds = concatenate_datasets(ds_list)
        else:
            self._ds = load_from_disk(ds_name_or_path)

        assert not isinstance(
            self._ds, DatasetDict
        ), "Dataset path should point to a single split, try adding /train ?."

        self._ds_total_length = len(self._ds)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        # This is where we shard the dataset into work for each dataparallel rank.
        # Our unit of work is now a "row" of the dataset though, not a file.

        self._worker_id = worker_id

        # max_num_rows = (len(self._ds) // num_shards) * num_shards
        max_num_rows = len(self._ds)
        index_list = list(range(shard_id, max_num_rows, num_shards))

        if index_list == []:
            self._ds_fingerprint = None
            self._ds_min = 0
            self._ds_max = 0
        else:
            self._ds_fingerprint = hashlib.shake_128(str(index_list).encode()).hexdigest(4)
            self._ds_min = min(index_list)
            self._ds_max = max(index_list)

        subds = self._ds.select(index_list)
        self._subds = subds

        self._ds_length = len(self._subds)

        logger.info(
            f"Rank {self._process_rank}/{self._num_processes}, worker {worker_id} has {self._ds_length}/{self._ds_total_length} rows | identifier={self._shortname}:{self._ds_fingerprint} | range={self._ds_min}:{self._ds_max} | head={index_list[:3]} | tail={index_list[-3:]}"
        )

        return HuggingfaceDatasetIterator(
            ds=subds,
            text_key=self._text_key,
            shortname=self._shortname,
            fingerprint=self._ds_fingerprint,
            worker_id=worker_id,
            process_rank=self._process_rank,
            num_processes=self._num_processes,
        )

    def __len__(self):
        return self._ds_length


class HuggingfaceDatasetIterator:
    def __init__(
        self,
        ds,
        text_key=None,
        shortname=None,
        fingerprint=None,
        worker_id=None,
        process_rank=None,
        num_processes=None,
    ):
        self._ds = ds
        self._text_key = text_key
        self._shortname = shortname
        self._ds_fingerprint = fingerprint
        self._worker_id = worker_id
        self._process_rank = process_rank
        self._num_processes = num_processes

        self._ds_iter = None

    def __len__(self):
        return len(self._ds)

    def __next__(self):
        if self._ds_iter is None:
            self._ds_iter = iter(self._ds)

        row = next(self._ds_iter)

        # this is the simplest possible operation, can add callables here that transform the row
        # to create the final string or strings that will be tokenized.
        row = row[self._text_key]
        return row


class HuggingfaceCombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None, data_telemetry=False):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        self._data_telemetry = data_telemetry
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets
        else:
            self._weights = [w / sum(weights) for w in weights]

        self.collate_fn = self._datasets[0]._collate_fn  # assumes all datasets have same collate_fn

    def __iter__(self):
        return HuggingfaceCombinedDatasetIterator(self._datasets, self._seed, self._weights, self._data_telemetry)


class HuggingfaceCombinedDatasetIterator:
    def __init__(self, datasets, seed, weights, data_telemetry=False):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        self._iter_ct = 0
        self._data_telemetry = data_telemetry

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        self._iter_ct += 1

        # this is the very beginning of data telemetry
        if self._data_telemetry and self._iter_ct < 5:
            logger.info(
                f"Draw result i={self._iter_ct} for rank={dataset._process_rank}/{dataset._num_processes}, worker={dataset._worker_id} | {dataset._shortname}:{dataset._ds_fingerprint}"
            )
        elif self._data_telemetry and self._iter_ct == 5:
            logger.info("Data telemetry off ...")

        return next(dataset)
