'''
Sample script

export OPENAI_API_KEY=<your_key>

python lit-gpt-dev/eval/factmem_rephrase.py --model tomg-group-umd/tinyllama_1b_redpajama_wiki2k_200B_tld3-step-00009536 --dataset "tomg-group-umd/RedPajama-Data-V2" --subset sample-100B --split train --num_samples 1000


'''

import time
import argparse
import os
import jsonlines
import json
from tqdm import tqdm
import torch

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

def str2bool(v):
    """Human friendly boolean cmdline flag parser."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected. Got: {str(v)}, " f"which cannot be converted to a boolean."
        )


def process_raw_data(raw_data, dataset):
    if dataset == "HuggingFaceTB/cosmopedia":
        return raw_data["prompt"] + raw_data["text"]
    elif dataset == "stingning/ultrachat":
        return "\n\n".join(raw_data["data"])
    elif dataset == "tomg-group-umd/RedPajama-Data-V2":
        return raw_data["raw_content"]
    else:
        try:
            return raw_data["text"]
        except:
            raise NotImplementedError(f"{dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, required=True)

    parser.add_argument("--dataset", default=None,required=True)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--min_length", default=64, type=int)
    parser.add_argument("--save_file_name", default=None, type=str)
    parser.add_argument("--seed", default=5, type=int)




    parser.add_argument("--dataset_type", default="huggingface")  # huggingface, huggingface_disk
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)

    parser.add_argument("--run_prelim_eval", type=str2bool, default=True)
    args = parser.parse_args()

    if args.save_file_name is None:
        args.save_file_name = f"rephrase_ppl_expts/rephrased/{args.num_samples}_{args.min_length}_{args.max_length}/rephrased.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token



    if args.dataset_type == "huggingface":
        raw_dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    elif args.dataset_type == "huggingface_disk":
        raw_dataset = datasets.load_from_disk(args.dataset)
    else:
        raise NotImplementedError(f"{args.dataset_type}")
    # TODO Add support for our hfds and pkds

    print(raw_dataset)
    raw_dataset_iterator = iter(raw_dataset)


    final_strings = []
    count = 0
    with torch.no_grad():
        with tqdm(total=args.num_samples) as pbar:
            while count < args.num_samples:
                torch.cuda.empty_cache()
                raw_data = next(raw_dataset_iterator)
                
                full_sequence = process_raw_data(raw_data, args.dataset)
                inputs = tokenizer(full_sequence, truncation=True, max_length=args.max_length, return_tensors="pt")
                
                if inputs.input_ids.shape[1] <= args.min_length or inputs.input_ids.shape[1] >= args.max_length :
                    continue
                else:
                   
                   final_strings.append(full_sequence) 

                pbar.update(1)
                count += 1



import openai

# Load your OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# List of strings you want to rephrase

def rephrase_strings(strings):
    rephrased = []
    for string in strings:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            max_tokens=args.max_length,
            messages=[
                {
                    "role": "user",
                    "content": f"Rephrase this sentence: {string}",
                },
            ],
        )
        rephrased_text = completion.choices[0].message.content

        rephrased.append({
            "original_text": string,
            "rephrased_text": rephrased_text
        })
    return rephrased

# Rephrase the strings
rephrased_strings = rephrase_strings(final_strings)

# Save the rephrased strings to a JSON file
os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
with jsonlines.open(args.save_file_name, "w") as linewriter:
    for row in tqdm(rephrased_strings):
        linewriter.write(row)

