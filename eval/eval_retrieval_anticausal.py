# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_from_disk, Dataset

from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.retrieval_model import PrefixSuffixNet
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint
from lit_gpt.multiple_negative_ranking_loss import cos_sim
from lit_gpt.retrieval_evaluator import RetrievalEvaluator


ROOT_DIR="/fs/cml-projects/llm-pretraining/llm-retrieval"
# data_dir = Path(f"{ROOT_DIR}/data/orca_retrieval")
# data_dir = Path(f"{ROOT_DIR}/data/openwebtext")
# data_dir = Path(f"{ROOT_DIR}/data/openwebtext_retrieval_val_data")
data_dir = Path(f"{ROOT_DIR}/data/orca_retrieval_val_data")
max_seq_length = None  # assign value to truncate
micro_batch_size = 1
doc_block_attn = True

@torch.no_grad()
def run_retrieval_metrics(fabric: L.Fabric, model: PrefixSuffixNet, val_data: Dataset, tokenizer: Tokenizer, attn_type: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    query_embeddings = []
    corpus_embeddings = []
    qrels = []
    for k in tqdm(range(len(val_data) // micro_batch_size), desc="Validating"):
        # ix = torch.arange(k * micro_batch_size, (k + 1) * micro_batch_size)
        ix = [i for i in range(k * micro_batch_size, (k + 1) * micro_batch_size) if i < len(val_data)]
        if len(ix) == 0:
            continue
        ix = torch.tensor(ix)
        # remove the indices of it's out of range
        # ix = ix[ix < len(val_data)]
        query_input_ids, corpus_input_ids = get_batch(fabric, val_data, ix=ix)
        # prefix_model_outputs, suffix_model_outputs = model(input_ids.to(fabric.device), output_hidden_states=True)
        query_outputs = model.prefix_model(query_input_ids, input_pos=None, attn_type="causal_attn", output_hidden_states=True)
        if attn_type == "anti_causal_attn":
            corpus_outputs = model.suffix_model(corpus_input_ids, input_pos=None, attn_type="anti_causal_attn", output_hidden_states=True)
        if attn_type == "causal_attn":
            reverse_idx = torch.flip(corpus_input_ids, [1])
            corpus_outputs = model.suffix_model(reverse_idx, input_pos=None, attn_type="causal_attn", output_hidden_states=True)
        query_hidden_states = query_outputs['hidden_states'][-1]    # (bsz, seq_len, d) picking the hidden states of the last layer
        corpus_hidden_states = corpus_outputs['hidden_states'][-1]    # (bsz, seq_len, d) picking the hidden states of the last layer
        # for query we want to pick the last token but need to make sure it's not a padding token so we use the query_len
        query_hidden_states = query_hidden_states[torch.arange(query_hidden_states.size(0)), torch.tensor(val_data[ix]['query_len']) - 1, :]        
        if attn_type == "causal_attn":
            # picking the last token's hidden states where attn_mask is all 1s
            corpus_hidden_states = corpus_hidden_states[:, -1, :]
        if attn_type == "anti_causal_attn":
            # picking the first token's hidden states for suffix_model where attn_mask is all 1s
            corpus_hidden_states = corpus_hidden_states[:, 0, :]
        query_embeddings.append(query_hidden_states.cpu())
        corpus_embeddings.append(corpus_hidden_states.cpu())
        qrels.extend(val_data[ix]['qrel'])
    query_embeddings = torch.cat(query_embeddings, dim=0)
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    # compute the cosine similarity between prefix and suffix
    similarity = cos_sim(query_embeddings, corpus_embeddings)
    prefix_suffix_ids = qrels
    id2arr_index = {id: i for i, id in enumerate(qrels)}

    predictions_top_100 = {i: [] for i in qrels}
    predictions_top_1 = {i: [] for i in qrels}
    for i in qrels:
        # find the top 1 similarity score
        top1 = torch.topk(similarity[id2arr_index[i]], k=1, largest=True, sorted=True)
        # find the top 1 similarity score's index
        top1_idx = top1.indices[0].item()
        # find the top 1 similarity score's id
        top1_id = qrels[top1_idx]
        # add the top 1 similarity score's id to the predictions
        predictions_top_1[i].append({"idx": top1_id, "score": top1.values[0].item()})
        # find the top 100 similarity scores
        top100 = torch.topk(similarity[id2arr_index[i]], k=100, largest=True, sorted=True)
        # find the top 100 similarity scores' indices
        top100_idxs = top100.indices
        # find the top 100 similarity scores' ids
        top100_ids = [qrels[top100_idx] for top100_idx in top100_idxs]
        # add the top 100 similarity scores' ids to the predictions
        predictions_top_100[i].extend([{"idx": top100_id, "score": top100_val.item()} for top100_id, top100_val in zip(top100_ids, top100.values)])
    # evaluator = RetrievalEvaluator([1], [1], [1], [1], [1], using_distances=False)   # only top 1s
    # from IPython import embed; embed()
    evaluator = RetrievalEvaluator(using_distances=False)
    qrels = {i: {i} for i in qrels}
    metrics = evaluator.compute_metrics(qrels, predictions_top_1)
    print("Top 1 metrics")
    print(metrics)
    metrics = evaluator.compute_metrics(qrels, predictions_top_100)
    print("Top 100 metrics")
    print(metrics)

@torch.inference_mode()
def main(
    data_dir: Path = data_dir,
    finetuned_path: Path = Path(f"{ROOT_DIR}/out/orca-retrieval-tiny-llama-1.1b/lit_model_finetuned.pth"),
    checkpoint_dir: Path = Path(f"{ROOT_DIR}/checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    precision: Optional[str] = None,
    max_seq_length: int = max_seq_length,
    attn_type: str = "anti_causal_attn",
    compile: bool = False,
) -> None:
    precision = precision or get_default_supported_precision(training=False)

    fabric = L.Fabric(devices=1, precision=precision)
    fabric.launch()
    # printing all the arguments
    fabric.print(locals())
    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")
    
    if max_seq_length is None or max_seq_length > config.block_size:
        max_seq_length = config.block_size

    checkpoint_path = finetuned_path

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = PrefixSuffixNet(config, checkpoint_dir)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()

    model = fabric.setup(model)
    tokenizer = Tokenizer(checkpoint_dir)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    
    val_data = load_from_disk(data_dir)
    # def split_corpus(example):
    #     np.random.seed(1234)
    #     query = example["query"]
    #     corpus = example["corpus"]
    #     # attaching the query to the corpus and then splitting the corpus into 2 parts to create new query and corpus
    #     corpus = query + corpus
    #     split = np.random.randint(5, len(corpus))
    #     new_query = corpus[:split]
    #     new_corpus = corpus[split:]
    #     # new_query = tokenizer.encode("Question: ", bos=False, eos=False).tolist() + query + tokenizer.encode(" Answer: ", bos=False, eos=False).tolist()
    #     # new_corpus = corpus
    #     return {"query": new_query, "corpus": new_corpus, "query_len": len(new_query), "corpus_len": len(new_corpus)}
    # val_data = val_data.map(split_corpus)
    # val_data = val_data.filter(lambda x: x['query_len'] <= max_seq_length and x['corpus_len'] <= max_seq_length)

    print("Number of examples in the validation data: ", len(val_data))
    # from IPython import embed; embed()
    val_data = val_data.add_column("id", range(len(val_data)))
    # train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    L.seed_everything(1234)
    run_retrieval_metrics(fabric, model, val_data, None, attn_type)
    fabric.print(f"Time to validate: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None, ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ix is None:
        ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    query_ids = [torch.tensor(data[i.item()]["query"]).type(torch.int64) for i in ix]
    corpus_ids = [torch.tensor(data[i.item()]["corpus"]).type(torch.int64) for i in ix]
    # ids = [data[i.item()]["id"] for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len_query = max(len(s) for s in query_ids)
    max_len_corpus = max(len(s) for s in corpus_ids)

    def pad_right(x, pad_id, max_len):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0, max_len=max_len_query) for x in query_ids])
    y = torch.stack([pad_right(x, pad_id=0, max_len=max_len_corpus) for x in corpus_ids])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x= fabric.to_device(x.pin_memory())
        y = fabric.to_device(y.pin_memory())
    else:
        x = fabric.to_device(x)
        y = fabric.to_device(y)

    return x, y


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
