import argparse
import os
import jsonlines
from tqdm import tqdm
import zlib
from pathlib import Path
import sys
import random

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.packed_cycle_dataset import PackedDataset


def process_raw_data(raw_data, dataset):
    if dataset == "HuggingFaceTB/cosmopedia":
        return raw_data["prompt"] + raw_data["text"]
    elif dataset == "togethercomputer/RedPajama-Data-1T":
        return raw_data["text"]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--local_files_only", default=False, type=bool)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_type", default="hfds")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=1000, type=int)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--save_file_name", default="mia_outputs/test.jsonl")

    parser.add_argument("--grad_norm", default=False, type=bool)
    parser.add_argument("--reference_model", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token

    if args.reference_model is not None:
        reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model).to(device)
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
        reference_tokenizer.pad_token = reference_tokenizer.eos_token

    ## set the models to eval mode
    model = model.eval()

    if args.dataset_type == "hfds":
        dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    elif args.dataset_type == "pkds":
        data_dir = args.dataset
        filenames = [str(pth) for pth in sorted(Path(data_dir).glob(f"{args.prefix}*"))]
        random.seed(args.seed)
        random.shuffle(filenames)

        if not filenames:
            raise FileNotFoundError(f"No files found at {str(data_dir)} with prefix {args.prefix}.")

        dataset = PackedDataset(filenames, n_chunks=4, block_size=args.max_length, shuffle=True, seed=args.seed)
    else:
        raise NotImplementedError

    dataset_iterator = iter(dataset)

    all_metrics = []
    for i in tqdm(range(args.start, args.end)):
        torch.cuda.empty_cache()
        raw_data = next(dataset_iterator)

        if args.dataset_type == "hfds":
            full_sequence = process_raw_data(raw_data, args.dataset)
            inputs = tokenizer(full_sequence, truncation=True, max_length=args.max_length, return_tensors="pt")
        elif args.dataset_type == "pkds":
            inputs = {"input_ids": raw_data.unsqueeze(0).to(device)}
            full_sequence = tokenizer.decode(raw_data)
        else:
            raise NotImplementedError

        for curr_ke in inputs:
            inputs[curr_ke] = inputs[curr_ke].to(device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        if args.grad_norm is True:
            outputs = model(**inputs)
            loss = outputs.loss.item()

            ### grad_norm
            grads = torch.autograd.grad(outputs.loss, model.parameters())
            grads_flattened = torch.cat([g.view(-1) for g in grads if g is not None])
            grad_norm = torch.norm(grads_flattened, p=2).item()
            curr_row = {"loss": loss, "grad_norm": grad_norm}
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss.item()
                curr_row = {"loss": loss}

        ### import zlib
        curr_row["zlip"] = loss / len(zlib.compress(bytes(full_sequence, "utf-8")))

        ### ref
        if args.reference_model is not None:
            with torch.no_grad():
                inputs = reference_tokenizer(
                    full_sequence, truncation=True, max_length=args.max_length, return_tensors="pt"
                )
                for curr_ke in inputs:
                    inputs[curr_ke] = inputs[curr_ke].to(device)
                labels = inputs["input_ids"].clone()
                labels[labels == reference_tokenizer.pad_token_id] = -100
                inputs["labels"] = labels

                outputs = reference_model(**inputs)
                curr_row["ref"] = loss - outputs.loss.item()

        all_metrics.append(curr_row)

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with jsonlines.open(args.save_file_name, "w") as linewriter:
        for row in tqdm(all_metrics):
            linewriter.write(row)
