# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import glob
import json
import os
import sys
from pathlib import Path
from multiprocessing import Process, cpu_count
import numpy as np
from tqdm import tqdm
import zstandard as zstd
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer
from lit_gpt.utils import CLI
# Starmap pool version
from multiprocessing import Pool
import time

filenames_sample = [
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
    "c4_sample.jsonl",
    "cc_2019-30_sample.jsonl",
    "cc_2020-05_sample.jsonl",
    "cc_2021-04_sample.jsonl",
    "cc_2022-05_sample.jsonl",
    "cc_2023-06_sample.jsonl",
    "github_sample.jsonl",
    "stackexchange_sample.jsonl",
    "wikipedia_sample.jsonl",
]

filename_sets = {
    "arxiv": "arxiv/arxiv*",
    "book": "book/book*",
    "c4": "c4/c4-train*",
    "common_crawl": "common_crawl/*/*",
    "github": "github/filtered*",
    "stackexchange": "stackexchange/stackexchange*",
    "wikipedia": "wikipedia/wiki*",
}


def prepare_sample(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()

def multi_prepare_files(subset, 
                        source_path: Path, 
                        tokenizer, 
                        is_cc: bool, 
                        destination_path: Path, 
                        chunk_size: int,
                        set_name: str,
                        process_id: int,
                        already_tokenized_files: list):
    filenames = subset
    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{set_name}_{process_id}",
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
    
    for name in filenames:
        filepath = source_path / name

        print(f"PID: {process_id} | Processing {filepath}", flush=True)
        if name in already_tokenized_files:
            print(f"PID: {process_id} | Skipping {name} as it is already tokenized.", flush=True)
            continue

        if is_cc:
            with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    text_ids = tokenizer.encode(text, eos=True)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                    
        else:
            with open(filepath, encoding="utf-8") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    text_ids = tokenizer.encode(text, eos=True)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
        with open('/lustre/orion/csc569/scratch/njain17/new_workspace/lit-gpt-dev/tokenize_redpajama_filenames_processed.txt', 'a') as f:
            print(name, file=f)
        
        print(f"PID: {process_id} | Finished processing {filepath}", flush=True)
    
    # builder.write_reminder()
    return True

def prepare_full(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = "",
    limit_file_num: int = -1, files_per_chunk_org: int = 2, skip_cc: bool = False, only_cc: bool = False,
    subset_prefix: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)
    args = []

    for set_name, pattern in filename_sets.items():
        if subset_prefix not in set_name and subset_prefix != "everything_else":
            print("Skipping", set_name)
            continue
        elif subset_prefix == "everything_else" and set_name not in ["book", "stackexchange", "wikipedia"]:
            print("Skipping", set_name)
            continue
        # else:

        print("Processing", set_name)
        print("Pattern", pattern)
        files_per_chunk = files_per_chunk_org
        if match and match not in set_name:
            continue

        is_cc = set_name == "common_crawl"

        assert not (skip_cc and only_cc), "Cannot skip and only include common crawl files."

        if is_cc and skip_cc:
            continue
        elif not is_cc and not skip_cc:
            continue

        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)
        # print("Filenames", filenames)
        if not filenames:
            raise RuntimeError(
                f"No files matching {pattern} found at {source_path}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )
    
        if limit_file_num > 0:
            if limit_file_num > len(filenames):
                limit_file_num = min(limit_file_num, len(filenames))
                print(f"Limiting to {limit_file_num} files.")
            filenames = filenames[:limit_file_num]

        

        # this chunks the work into huge file lists for each single process to complete
        # chunked_filenames = np.array_split(filenames, num_processes)

        # instead, we can pick a max work unit by num files and cull each process more frequently
        if len(filenames) < files_per_chunk:
            files_per_chunk = len(filenames)
        num_chunks = len(filenames) // files_per_chunk
        chunked_filenames = np.array_split(filenames, num_chunks)

        print(
            f"Running in files per chunk mode, with fpc={files_per_chunk} for {len(chunked_filenames)} total chunks.",
            flush=True,
        )
        already_tokenized_file_path = '/lustre/orion/csc569/scratch/njain17/new_workspace/lit-gpt-dev/tokenize_redpajama_filenames_processed.txt'
        if os.path.exists(already_tokenized_file_path):
            with open(already_tokenized_file_path, 'r') as f:
                already_tokenized_files = f.readlines()
            already_tokenized_files = [x.strip() for x in already_tokenized_files]
        else:
            already_tokenized_files = []
        print(f"Total files: {len(already_tokenized_files)}")
        # print(f"Total files: {already_tokenized_files}")



        start_time = time.time()
        # prepare args for starmap
        for process_id, subset in enumerate(chunked_filenames):
            # print(f"PID: {process_id} | Subset length: {len(subset)}", flush=True)
            # print(f"Files: {subset}", flush=True)
            args.append(    
                    (
                    subset, 
                    source_path, 
                    tokenizer, 
                    is_cc, 
                    destination_path, 
                    chunk_size,
                    set_name,
                    process_id,
                    already_tokenized_files
                    )
            )
    # node_id = os.environ["SLURM_PROCID"]
    # node_id = int(node_id)
    # # We will split the args based on the PROCID across all nodes
    # print(len(args))
    # print(int(os.environ["SLURM_JOB_NUM_NODES"]))
    # chunk_size_per_node = len(args) // int(os.environ["SLURM_JOB_NUM_NODES"])
    # print("Node ID:", node_id)
    # print("Total Args length:", len(args))
    # print("Args length:", len(args[node_id*chunk_size_per_node:(node_id+1)*chunk_size_per_node]))
    # args = args[node_id*chunk_size_per_node:(node_id+1)*chunk_size_per_node]
    print("Filename of first filename:", args[0][0][0])
    # You get the args based on your PROCID
    num_processes = min(cpu_count(), len(args))
    print(f"Total Args length: {len(args)}")
    print(f"Total of {num_processes} processes")
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(multi_prepare_files, args)

    print(f"Results w/ True ret val: {sum(results)/len(results)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


        # for name in filenames:
        #     filepath = source_path / name

        #     print(f"Processing {name}")

        #     if is_cc:
        #         with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        #             for row in tqdm(f):
        #                 text = json.loads(row)["text"]
        #                 text_ids = tokenizer.encode(text, eos=True)
        #                 builder.add_array(np.array(text_ids, dtype=builder.dtype))
                        
        #     else:
        #         with open(filepath, encoding="utf-8") as f:
        #             for row in tqdm(f):
        #                 text = json.loads(row)["text"]
        #                 text_ids = tokenizer.encode(text, eos=True)
        #                 builder.add_array(np.array(text_ids, dtype=builder.dtype))
        # builder.write_reminder()


def prepare(
    source_path: Path = Path("/lustre/orion/csc569/proj-shared/language_datasets/raw/RedPajama1T"),
    checkpoint_dir: Path = Path("/lustre/orion/csc569/proj-shared/language_models/external/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    destination_path: Path = Path("/lustre/orion/csc569/proj-shared/language_datasets/processed/redpajama1T"),
    sample: bool = False,
    match: str = "",
    skip_cc: bool = False,
    only_cc: bool = False,
    files_per_chunk: int = 2,
    subset_prefix: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
    config = Config.from_checkpoint(checkpoint_dir)

    prepare_fn = prepare_sample if sample else prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(config.block_size + 1) * 512,  # block size + 1 for causal, 1024 blocks
        match=match,
        skip_cc=skip_cc,
        only_cc=only_cc,
        files_per_chunk_org=files_per_chunk,
        subset_prefix=subset_prefix
    )


if __name__ == "__main__":
    CLI(prepare)
