import time
import argparse
import os
import jsonlines
import json
from tqdm import tqdm

import datasets
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
    else:
        try:
            return raw_data["text"]
        except:
            raise NotImplementedError(f"{dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_strategy", default="device")  # device, device_map
    parser.add_argument("--device_map", default="auto")  # auto, balanced, balanced_low_0, sequential
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_type", default="huggingface")  # huggingface, huggingface_disk
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--save_file_name", default="data_extraction_outputs/test.jsonl", type=str)
    parser.add_argument("--inference_len", default=50, type=int)
    parser.add_argument("--run_prelim_eval", type=str2bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")

    if args.model_strategy == "device":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
        model.to(device)
        print(model.device)
    elif args.model_strategy == "device_map":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=args.device_map,
            torch_dtype=torch.bfloat16,
        )
        print(json.dumps(model.hf_device_map, indent=4))
    else:
        raise NotImplementedError(f"model_strategy: {args.model_strategy}")

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    ## set the models to eval mode
    model = model.eval()

    if args.dataset_type == "huggingface":
        raw_dataset = datasets.load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    elif args.dataset_type == "huggingface_disk":
        raw_dataset = datasets.load_from_disk(args.dataset)
    else:
        raise NotImplementedError(f"{args.dataset_type}")
    # TODO Add support for our hfds and pkds

    print(raw_dataset)

    raw_dataset_iterator = iter(raw_dataset)

    generate_kwargs = dict(
        do_sample=False,
        num_return_sequences=1,
        max_new_tokens=args.inference_len,
    )

    gen_times = []
    gen_token_cts = []

    all_gens = []
    count = 0
    with torch.no_grad():
        with tqdm(total=args.num_samples) as pbar:
            while count < args.num_samples:
                torch.cuda.empty_cache()
                raw_data = next(raw_dataset_iterator)
                full_sequence = process_raw_data(raw_data, args.dataset)
                inputs = tokenizer(full_sequence, truncation=True, max_length=args.max_length, return_tensors="pt")

                if inputs.input_ids.shape[1] <= args.inference_len:
                    continue

                for curr_ke in inputs:
                    inputs[curr_ke] = inputs[curr_ke].to(device)

                gt_ids = inputs.input_ids[:, -args.inference_len :]
                input_ids = inputs.input_ids[:, : -args.inference_len]

                t0 = time.perf_counter()
                model_output = model.generate(input_ids=input_ids, **generate_kwargs)
                t1 = time.perf_counter()
                gen_time = t1 - t0
                gen_times.append(gen_time)

                model_output = model_output[:, input_ids.shape[-1] :].cpu()

                gen_token_cts.append(model_output.shape[1])

                decoded_output = tokenizer.batch_decode(
                    model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                decoded_input = tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                decoded_gt = tokenizer.batch_decode(
                    gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                all_gens.append(
                    {
                        "original_prompt": full_sequence,
                        "test_input": decoded_input[0],
                        "gt_output": decoded_gt[0],
                        "gen_output": decoded_output[0],
                        "gt_output_ids": gt_ids.cpu().squeeze().tolist(),
                        "gen_output_ids": model_output.cpu().squeeze().tolist(),
                    }
                )

                # wait till hot then show output and time
                if count == 4 or (count == args.num_samples - 1):
                    print(f"{'#'*80}\nInput:\n{decoded_input[0]}")
                    print(f"{'-'*80}\nGenerated Output:\n{decoded_output[0]}")
                    print(f"{'-'*80}\nReference Output:\n{decoded_gt[0]}")
                    print(f"Time to generate: {gen_time :.02f} seconds.")

                pbar.update(1)
                count += 1

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with jsonlines.open(args.save_file_name, "w") as linewriter:
        for row in tqdm(all_gens):
            linewriter.write(row)

    # Report stats (skip the first one as it is usually slower)
    if torch.cuda.is_available():
        print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    print(f"Average tokens generated per inference: {sum(gen_token_cts) / len(gen_token_cts):.02f} tokens.")
    if len(gen_times) > 1:
        print(f"Average time to generate: {sum(gen_times[1:]) / len(gen_times[1:]):.02f} seconds.")
        print(f"Average tokens generated per second: {sum(gen_token_cts[1:]) / sum(gen_times[1:]):.02f} tokens/sec.")

    if not args.run_prelim_eval:
        exit()
    ### prelim eval
    rouge = evaluate.load("rouge")

    exact_match = 0
    rouge_metrics = {}

    for curr_data in tqdm(all_gens):
        if curr_data["gt_output_ids"] == curr_data["gen_output_ids"]:
            exact_match += 1

        predictions = [curr_data["gen_output"]]
        references = [curr_data["gt_output"]]
        results = rouge.compute(predictions=predictions, references=references)

        for key in results:
            if key not in rouge_metrics:
                rouge_metrics[key] = results[key]
            else:
                rouge_metrics[key] += results[key]

    exact_match_rate = exact_match / len(all_gens)
    for key in rouge_metrics:
        rouge_metrics[key] = rouge_metrics[key] / len(all_gens)

    print(f"save_file_name: {args.save_file_name}")
    print(f"exact_match_rate: {exact_match_rate}")
    print("rouge_metrics")
    print(rouge_metrics)
