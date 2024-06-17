# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional, Union

import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.bos_id = None
        self.eos_id = None
        self.pad_id = None

        if (checkpoint_dir / "tokenizer.json").is_file():
            from transformers import AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=False, add_eos_token=False
            )

            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                self.bos_id = self.processor.bos_token_id
                self.eos_id = self.processor.eos_token_id
                self.pad_id = self.processor.pad_token_id
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        elif "open_llama" in str(checkpoint_dir):
            from transformers import LlamaTokenizer

            self.processor = LlamaTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=False, add_eos_token=False
            )

            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                self.bos_id = self.processor.bos_token_id
                self.eos_id = self.processor.eos_token_id
                self.pad_id = self.processor.pad_token_id
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        else:
            raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)

        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
