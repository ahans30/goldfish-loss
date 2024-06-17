'''
Sample script

python lit-gpt-dev/eval/mauve_comp.py --model tomg-group-umd/tinyllama_1b_redpajama_wiki2k_200B_tld3-step-00009536 --dataset tomg-group-umd/RedPajama-Data-V2 --subset sample-100B --split train --inference_len 128 --sampling_type greedy --num_samples 500
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
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--min_length", default=512, type=int)
    parser.add_argument("--save_file_name", default=None, type=str)
    parser.add_argument("--inference_len", default=50, type=int)
    parser.add_argument("--seed", default=5, type=int)

    parser.add_argument("--sampling_type", default="greedy", type=str)
    parser.add_argument("--use_wandb", default="True", type=str)


    parser.add_argument("--model_strategy", default="device")  # device, device_map
    parser.add_argument("--device_map", default="auto")  # auto, balanced, balanced_low_0, sequential
    parser.add_argument("--dataset_type", default="huggingface")  # huggingface, huggingface_disk
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)

    parser.add_argument("--run_prelim_eval", type=str2bool, default=True)
    args = parser.parse_args()

    if args.save_file_name is None:
        args.save_file_name = f"mauve_eval_outputs/{args.model.split('/')[-1]}/{args.inference_len}_{args.sampling_type}/{args.min_length}_{args.max_length}/{args.num_samples}_{args.seed}.jsonl"

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

    ## create file name if its none

    if args.use_wandb == "True":
        import wandb
        wandb.init(project="tld_mauve",name=f"{args.save_file_name.split('mauve_eval_outputs/')[-1]}")
        wandb.config.update(args)

    else:
        wandb = None
        
    set_seed(args.seed)

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


    if args.sampling_type == "greedy":
        generate_kwargs = dict(
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=args.inference_len,
        )
    elif args.sampling_type == "ancestral":
        generate_kwargs = dict(
            do_sample=True , num_beams=1,
            num_return_sequences=1,
            max_new_tokens=args.inference_len,
        )
    elif args.sampling_type == "nucleus":
        generate_kwargs = dict(
            do_sample=True , 
            top_p=0.92,
            top_k=0,
            num_return_sequences=1,
            max_new_tokens=args.inference_len,
        )


    gen_times = []
    gen_token_cts = []

    all_gens = []
    og_text_list = []
    gen_text_list = []
    count = 0
    with torch.no_grad():
        with tqdm(total=args.num_samples) as pbar:
            while count < args.num_samples:
                torch.cuda.empty_cache()
                raw_data = next(raw_dataset_iterator)
                
                full_sequence = process_raw_data(raw_data, args.dataset)
                inputs = tokenizer(full_sequence, truncation=True, max_length=args.max_length, return_tensors="pt")

                if inputs.input_ids.shape[1] <= max(args.inference_len, args.min_length):
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
                if len(model_output[0])< 5:
                    continue
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
                
                og_text_list.append(decoded_gt[0])
                gen_text_list.append(decoded_output[0])
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
    
    import mauve 

    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(q_text=gen_text_list, p_text=og_text_list, device_id=0, max_text_length=args.inference_len, verbose=False)
    print(out.mauve) 
    wandb.log({
            'mauve_score': out.mauve
        })




