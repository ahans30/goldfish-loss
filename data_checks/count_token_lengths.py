#### This is a script to run the overlap test on the two datasets where one is way smaller than the other ####
import os
import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def count_tokenizes_and_get_metrics(dataset, tokenizer):
    token_counts = []
    for data in tqdm(dataset, total=len(dataset)):
        token_counts.append(len(tokenizer(data['text'])['input_ids']))
    # we gonnna return the mean, median, max, min, and std as a string
    return f"Mean: {np.mean(token_counts)}, Median: {np.median(token_counts)}, Max: {np.max(token_counts)}, Min: {np.min(token_counts)}, Std: {np.std(token_counts)}"
    print("Mean: ", np.mean(token_counts))
    print("Median: ", np.median(token_counts))
    print("Max: ", np.max(token_counts))
    print("Min: ", np.min(token_counts))
    print("Std: ", np.std(token_counts))

    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--num_proc", type=int, default=28)
    args = parser.parse_args()

    for file in os.listdir(args.base_dir):
        if file != 'non_targeted':
            for split in ['wiki', 'random']:
                new_path = os.path.join(args.base_dir, file, split)
                if os.path.exists(new_path):
                    print("Processing: ", new_path)
                    dataset = datasets.load_from_disk(new_path)
                    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
                    output=count_tokenizes_and_get_metrics(dataset, tokenizer)
                    print(output)
        elif file == 'non_targeted':
            new_path = os.path.join(args.base_dir, file)
            if os.path.exists(new_path):
                print("Processing: ", f"{new_path}/wiki")
                dataset = datasets.load_from_disk(f"{new_path}/wiki")
                tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
                output=count_tokenizes_and_get_metrics(dataset, tokenizer)
                print(output)
        else:
            raise ValueError("Invalid File: ", file)
        # print("File: ", file)



