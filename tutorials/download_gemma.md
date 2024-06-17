## Download [Gemma](https://blog.google/technology/developers/gemma-open-models/) weights

Google developed and publicly released the Gemma large language models (LLMs), a collection of pretrained models in 2B and 7B parameter size that are based on the Gemini architecture.

For more information, please see the [technical report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf).


To see all the available checkpoints, run:

```bash
python scripts/download.py | grep gemma
```

which will print

```text
google/gemma-7b
google/gemma-2b
google/gemma-7b-it
google/gemma-2b-it
```

In the list above, `gemma-2b` and `gemma-7b` are the pretrained models, and `gemma-2b-it` and `gemma-7b-it` are the instruction-finetuned models.

In order to use a specific checkpoint, for instance [gemma-2b](https://huggingface.co/google/gemma-2b), download the weights and convert the checkpoint to the lit-gpt format.

This requires that you've been granted access to the weights on the HuggingFace hub. You can do so by following the steps at <https://huggingface.co/google/gemma-2b>.
After access is granted, you can find your HF hub token in <https://huggingface.co/settings/tokens>.

```bash
pip install 'huggingface_hub[hf_transfer] @ git+https://github.com/huggingface/huggingface_hub'

python scripts/download.py --repo_id google/gemma-2b --access_token your_hf_token --from_safetensors true

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/google/gemma-2b
```

By default, the `convert_hf_checkpoint` step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
python chat/base.py --checkpoint_dir checkpoints/google/gemma-2b
```
