# Toy LLM

A small academic character-level GPT built with Python and PyTorch. It implements the core pieces of a local language model: tokenizer, dataset batching, causal self-attention, Transformer blocks, training, checkpointing, and sampling.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Train

Put a UTF-8 text corpus at `data/input.txt`, then run:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Checkpoints are written to `checkpoints/latest.pt`.

To train from a list of URLs, put one URL per line in `data/my_links_corpus.txt` and run:

```bash
python scripts/train.py --config configs/links.yaml
```

`dataset_kind: url_list` makes the trainer fetch each URL, strip basic HTML into readable text, cache the combined corpus at `data/fetched_links_corpus.txt`, and train from that cache on later runs. Delete the cache file when you want to re-fetch the URLs.

To resume training:

```bash
python scripts/train.py --config configs/tiny.yaml --resume checkpoints/latest.pt
```

## Generate

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --max-new-tokens 300
```

Generation options:

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --temperature 0.8 --top-k 40
```

Prompts can only contain characters seen in the training corpus. Version 1 raises a clear error for unknown characters.

## Inspect Tokenizer

```bash
python scripts/inspect_tokenizer.py --dataset data/input.txt
```

For a URL list:

```bash
python scripts/inspect_tokenizer.py --dataset data/my_links_corpus.txt --url-list --cache data/fetched_links_corpus.txt
```

## Tests

```bash
pytest
```

## Project Layout

```text
toy_llm/
  config.py       YAML-backed dataclass config
  tokenizer.py    deterministic character tokenizer
  dataset.py      UTF-8 loading and random batch sampling
  model.py        tiny GPT-style decoder-only Transformer
  train_loop.py   evaluation, training, checkpoint scheduling
  checkpoint.py   save/load helpers
  device.py       auto CUDA/MPS/CPU selection
  sampling.py     small model utilities
```
