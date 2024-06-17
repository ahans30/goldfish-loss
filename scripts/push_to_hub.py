import torch
import transformers
from huggingface_hub import create_repo

transformers.logging.set_verbosity_info()


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    # parser.add_argument("--repo_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--token_id", type=str, required=True)

    args = parser.parse_args()
    args.repo_name = f"tomg-group-umd/{args.model_name}"
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path)
    state_dict = torch.load(f"{args.model_path}/pytorch_model.bin")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, state_dict=state_dict)
    print(model)

    create_repo(args.repo_name, private=True, token=args.token_id, exist_ok=True)
    model.push_to_hub(args.repo_name, use_temp_dir=True, token=args.token_id, overwrite=True)
    tokenizer.push_to_hub(args.repo_name, use_temp_dir=True, token=args.token_id)

    print(f"Model pushed to {model}")
