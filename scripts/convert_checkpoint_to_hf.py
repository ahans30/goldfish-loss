import json
import shutil
import sys
from pathlib import Path

import torch

from convert_pretrained_checkpoint import convert_checkpoint
from convert_lit_checkpoint import convert_lit_checkpoint

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import create_repo

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.utils import CLI


@torch.inference_mode()
def convert_checkpoint_to_hf(
    checkpoint_file: Path,
    tokenizer_dir: Path,
    model_name: str,
    axonn_patch: bool = False,
    push_to_hub: bool = True,
) -> None:
    ### convert training checkpoint to lit checkpoint
    parent_dir = checkpoint_file.parent.absolute()
    with open(parent_dir / "model_config.json") as f:
        model_config = json.load(f)
    config_name = model_config["name"]
    convert_checkpoint(checkpoint_file, tokenizer_dir, config_name, parent_dir / f"lit_checkpoint_{model_name}")

    ### convert training checkpoint to hf checkpoint
    convert_lit_checkpoint(
        parent_dir / f"lit_checkpoint_{model_name}/lit_model.pth",
        parent_dir / f"hf_checkpoint_{model_name}/pytorch_model.bin",
        parent_dir / f"lit_checkpoint_{model_name}/lit_config.json",
        axonn_patch=axonn_patch,
    )

    for tokenizer_file in tokenizer_dir.glob("tokenizer*"):
        shutil.copyfile(tokenizer_file, parent_dir / f"hf_checkpoint_{model_name}" / tokenizer_file.name)

    if (tokenizer_dir / "generation_config.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "generation_config.json",
            parent_dir / f"hf_checkpoint_{model_name}" / "generation_config.json",
        )

    if (tokenizer_dir / "special_tokens_map.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "special_tokens_map.json",
            parent_dir / f"hf_checkpoint_{model_name}" / "special_tokens_map.json",
        )

    if (tokenizer_dir / "added_tokens.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "added_tokens.json", parent_dir / f"hf_checkpoint_{model_name}" / "added_tokens.json"
        )

    if (tokenizer_dir / "config.json").is_file():
        shutil.copyfile(tokenizer_dir / "config.json", parent_dir / f"hf_checkpoint_{model_name}" / "config.json")

    # hf_org = model_config["hf_config"]["org"]
    # hf_name = model_config["hf_config"]["name"]
    # hf_config = AutoConfig.from_pretrained(f"{hf_org}/{hf_name}")
    # hf_config = hf_config.to_dict()
    # with open(parent_dir / f"hf_checkpoint_{model_name}" / "config.json", "w") as f:
    #     json.dump(hf_config, f, indent=4)

    if not push_to_hub:
        return

    ### push to hub
    repo_name = f"tomg-group-umd/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(parent_dir / f"hf_checkpoint_{model_name}")
    state_dict = torch.load(parent_dir / f"hf_checkpoint_{model_name}/pytorch_model.bin")
    model = AutoModelForCausalLM.from_pretrained(parent_dir / f"hf_checkpoint_{model_name}", state_dict=state_dict)
    create_repo(repo_name, private=True)
    model.push_to_hub(repo_name, use_temp_dir=True)
    tokenizer.push_to_hub(repo_name, use_temp_dir=True)

    print(f"Model pushed to {repo_name}")


if __name__ == "__main__":
    CLI(convert_checkpoint_to_hf)
