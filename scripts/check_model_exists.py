import torch
import transformers
import sys
from huggingface_hub import delete_repo
import os

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()

    try:
        model = AutoModel.from_pretrained(f"tomg-group-umd/{args.model_name}")
        print(f"Repo {args.model_name} exists")
        sys.exit(0)
    except Exception as e:
        try:
            delete_repo(repo_id = args.model_name, token = os.environ["HF_TOKEN_WRITE"])
        except Exception as e:
            pass
        print(f"Repo {args.model_name} does NOT exist")
        sys.exit(1)