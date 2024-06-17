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
from lit_gpt.data_scheduler_utils import DataSchedulerTracker

import logging

logger = logging.getLogger(__name__)

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self,
        filenames,
        n_chunks,
        block_size,
        seed=12345,
        shuffle=True,
        wrap=False,
        num_processes=1,
        process_rank=0,
        shortname=None,
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._ds_fingerprint = None
        self._shortname = shortname  # This is human readble, correps to the full file list.

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        total_num_files = len(self._filenames)
        max_num_files = total_num_files // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        self._ds_fingerprint = hashlib.shake_128(str(filenames).encode()).hexdigest(
            4
        )  # This is not human readable, corresp to the file list _this_ process is using.

        logger.info(
            f"Rank {self._process_rank}/{self._num_processes}, worker {worker_id} has {len(filenames)}/{total_num_files} files | identifier={self._shortname}:{self._ds_fingerprint}"
        )

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            shortname=self._shortname,
            fingerprint=self._ds_fingerprint,
            worker_id=worker_id,
            process_rank=self._process_rank,
            num_processes=self._num_processes,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []
        self._total_tokens_exact = 0
        self._filler_sep_tokens = 0

    def _write_chunk(self, skip_write=False):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        # right before we write, we can compute the number of tokens being written
        # and update the total number of tokens
        last_non_sep_idx = np.argwhere((self._arr != self._sep_token)).squeeze()[-1]
        tokens_in_chunk = last_non_sep_idx + 1  # +1 for zero-indexing

        if skip_write:
            self._arr.fill(self._sep_token)
            self._idx = 0
            return tokens_in_chunk  # amount we are skipping

        self._filler_sep_tokens += self._chunk_size - tokens_in_chunk
        self._total_tokens_exact += tokens_in_chunk
        # print(
        #     f"Chunk written with {tokens_in_chunk} tokens and {self._filler_sep_tokens} filler sep tokens"
        # )

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_remainder(self):
        self._write_chunk()

    def skip_write_remainder(self):
        return self._write_chunk(skip_write=True)


class PackedDatasetIterator:
    def __init__(
        self,
        filenames,
        n_chunks,
        block_size,
        seed,
        shuffle,
        wrap,
        shortname=None,
        fingerprint=None,
        worker_id=None,
        process_rank=None,
        num_processes=None,
    ):
        self._shortname = shortname
        self._ds_fingerprint = fingerprint
        self._worker_id = worker_id
        self._process_rank = process_rank
        self._num_processes = num_processes

        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            if not self._wrap:
                raise StopIteration
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, data_scheduler_tracker=None, data_telemetry=False):
        self._seed = seed
        self._datasets = datasets
        self._data_scheduler_tracker = data_scheduler_tracker
        self._data_telemetry = data_telemetry
        n_datasets = len(datasets)
        if data_scheduler_tracker is None:
            self._data_scheduler_tracker = DataSchedulerTracker([1 / n_datasets] * n_datasets)

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._data_scheduler_tracker, self._data_telemetry)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, data_scheduler_tracker, data_telemetry=False):
        self._datasets = datasets
        self._datasets_iterators = [iter(el) for el in datasets]
        self._num_datasets = len(datasets)
        self._data_scheduler_tracker = data_scheduler_tracker
        self._rng = random.Random(seed)
        self._iter_ct = 0
        self._data_telemetry = data_telemetry

    def __next__(self):
        if sum(self._data_scheduler_tracker.weights) == 0:
            if self._data_scheduler_tracker.base_id is not None:
                # if all buckets have 0 weight, return the base dataset
                self._data_scheduler_tracker.weights[self._data_scheduler_tracker.base_id] = 100
                return self.__next__()
            else:
                # if all buckets have 0 weight and no base dataset, return empty
                return torch.tensor([])

        (dataset_idx,) = self._rng.choices(range(self._num_datasets), weights=self._data_scheduler_tracker.weights, k=1)
        dataset = self._datasets_iterators[dataset_idx]

        try:
            curr_data = next(dataset)
            self._data_scheduler_tracker.sample_count[dataset_idx] += 1

            self._iter_ct += 1

            # this is the very beginning of data telemetry
            if self._data_telemetry and self._iter_ct < 5:
                logger.info(
                    f"Draw result i={self._iter_ct} for rank={dataset._process_rank}/{dataset._num_processes}, worker={dataset._worker_id} | {dataset._shortname}:{dataset._ds_fingerprint}"
                )
            elif self._data_telemetry and self._iter_ct == 5:
                logger.info("Data telemetry off ...")

            return curr_data
        except:
            self._data_scheduler_tracker.epoch_count[dataset_idx] += 1
            self._datasets_iterators[dataset_idx] = iter(self._datasets[dataset_idx])

            if (self._data_scheduler_tracker.max_epochs is not None) and (
                self._data_scheduler_tracker.max_epochs[dataset_idx]
                <= self._data_scheduler_tracker.epoch_count[dataset_idx]
            ):
                # if exceeds max epoch
                self._data_scheduler_tracker.weights[dataset_idx] = 0
                return self.__next__()
            else:
                dataset = self._datasets_iterators[dataset_idx]
                curr_data = next(dataset)
                self._data_scheduler_tracker.sample_count[dataset_idx] += 1

                return curr_data
