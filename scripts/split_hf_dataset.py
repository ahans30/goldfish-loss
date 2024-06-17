# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import json
import os
import sys
from pathlib import Path
from typing import Union

from datasets import load_dataset, DownloadConfig, concatenate_datasets, load_from_disk  # huggingface datasets

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer
import os
# os.environ['CURL_CA_BUNDLE'] = ''
# from datasets import clear_cache
# clear_cache()

def prepare(
    destination_path: Path = Path("/fs/cml-projects/llm-pretraining/llm-retrieval/data/splitted_radpajama_v2_10B"),
    seed: int = 42,
    cache_dir: Path = Path("/fs/cml-projects/llm-pretraining/llm-retrieval/data/cache"),
    test_size: Union[float, int, None] = 0.0005,
) -> None:
    

    destination_path.mkdir(parents=True, exist_ok=True)

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = os.cpu_count() // 2

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on HW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc
    dl_config = DownloadConfig(max_retries=5)
    # num_proc_load_dataset = 2
    # from IPython import embed; embed()
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    # dataset = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample-10B", languages=["en"], download_config=dl_config, trust_remote_code=True, num_proc=num_proc_load_dataset, cache_dir=cache_dir)
    
    subset_names = ['auto_math_text', 'khanacademy', 'openstax', 'stanford', 'stories', 'web_samples_v1', 'web_samples_v2', 'wikihow']
    datasets = []

    for subset in subset_names:
        dataset = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", num_proc=num_proc_load_dataset, cache_dir=cache_dir)
        dataset = dataset.add_column("subset", [subset] * len(dataset))
        datasets.append(dataset)
    combined_dataset = concatenate_datasets(datasets)
    combined_dataset.save_to_disk("/fs/cml-projects/llm-pretraining/llm-retrieval/data/cosmopedia/")
    dataset = load_from_disk("/fs/cml-projects/llm-pretraining/datasets/raw/cosmopedia/")
    # dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, cache_dir=cache_dir, split="train")
    # remiving all columns except text

    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    test_size = 10000 / len(dataset) # picking 5000 samples for test set
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val_ood"] = split_dataset.pop("test")  # rename the test split to val_ood
    test_size = 10000 / len(split_dataset['train']) # picking 5000 samples for test set
    second_split_dataset = split_dataset['train'].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val_id"] = second_split_dataset.pop("test")
    split_dataset["train"] = second_split_dataset.pop("train")
    # split_dataset['train'].save_to_disk('/fs/cml-projects/llm-pretraining/llm-retrieval/data/cosmopedia/train')
    split_dataset.save_to_disk('/fs/cml-projects/llm-pretraining/llm-retrieval/data/new_splitted_cosmopedia', num_proc=num_proc)
    # split_dataset.save_to_disk('/fs/cml-projects/llm-pretraining/llm-retrieval/data/splitted_openwebtext', num_proc=num_proc)


    # def process(example):
    #     ids = tokenizer.encode(example["text"]).tolist()
    #     ids.append(tokenizer.eos_id)
    #     # ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    #     # ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    #     # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    #     return {"ids": ids, "len": len(ids)}

    # # tokenize the dataset
    # tokenized = split_dataset.map(process, remove_columns=["text"], desc="tokenizing the splits", num_proc=num_proc)

    # # concatenate all the ids in each dataset into one large file we can use for training
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset["len"], dtype=np.uint64)
    #     filename = destination_path / f"{split}.bin"
    #     dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    #     arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
    #     total_batches = 1024

    #     idx = 0
    #     for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
    #         # Batch together samples for faster write
    #         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
    #         arr_batch = np.concatenate(batch["ids"])
    #         # Write into mmap
    #         arr[idx : idx + len(arr_batch)] = arr_batch
    #         idx += len(arr_batch)
    #     arr.flush()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)