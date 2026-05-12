<p align="center">
  <img src="assets/vibroso-logo.svg" alt="Vibroso logo" width="720">
</p>

# Vibroso

A small academic GPT-style language model built with Python and PyTorch. It implements the core pieces of a local language model: tokenization, dataset batching, batched causal self-attention with RoPE and KV-cache generation, RMSNorm Transformer blocks, training, checkpointing, and sampling.

## What this is

Vibroso is a from-scratch educational GPT-style language model.

## What this is not

It is not a production LLM, chatbot, or instruction-following assistant.

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

The default tokenizer is a small trainable byte-level BPE tokenizer. It starts with all 256 byte values, learns frequent adjacent byte-token merges from the training corpus, and stores the learned merges in the checkpoint. Use `tokenizer_kind: char` in a config if you want the original character-level behavior, or `tokenizer_kind: sentencepiece` for the optimized SentencePiece backend.

After tokenizer changes, retrain from scratch instead of resuming older subword checkpoints. Old subword tokenizer payloads are rejected so model weights are not reused with incompatible token IDs.

The Transformer MLP is configurable with `model.mlp_type: gelu` or `model.mlp_type: swiglu`. GELU remains the default for old configs and checkpoints. SwiGLU changes model weights and requires training a new checkpoint with `mlp_type: swiglu`.

MLP options:

```yaml
model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.1
  mlp_type: gelu    # default baseline
```

```yaml
model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.1
  mlp_type: swiglu  # gated MLP, train from scratch
```

For a full SwiGLU example, use `configs/sentencepiece_swiglu.yaml`.

For large corpora, `tokenizer_train_chars` controls how many leading corpus characters are used to learn BPE merges. The base byte vocabulary still covers any valid UTF-8 text. The Wikipedia config starts with `tokenizer_vocab_size: 2048` and `tokenizer_train_chars: 10000`; increase the training window only after confirming startup time is acceptable.

Wikipedia training has three startup phases before model steps begin: fetch/cache pages, build the tokenizer, then encode the cached corpus. The trainer prints `building tokenizer...` and `encoding dataset...` so you can tell which phase is running.

On CUDA devices, training uses bf16 mixed precision with PyTorch autocast and GradScaler. CPU and MPS training leave autocast disabled.

To train from a list of URLs, put one URL per line in `data/my_links_corpus.txt` and run:

```bash
python scripts/train.py --config configs/links.yaml
```

`dataset_kind: url_list` makes the trainer fetch each URL, strip basic HTML into readable text, cache the combined corpus at `data/fetched_links_corpus.txt`, and train from that cache on later runs. Delete the cache file when you want to re-fetch the URLs.

To train from Wikipedia page titles, put one title per line in `data/my_wikipedia_pages.txt`:

```text
Artificial intelligence
Language model
Transformer (deep learning architecture)
```

Then run:

```bash
python scripts/train.py --config configs/wikipedia.yaml
```

`dataset_kind: wikipedia_titles` uses the `wikipedia-api` package, calls `page(title).text` for each listed title, caches the combined text at `data/fetched_wikipedia_corpus.txt`, and trains from that cache on later runs.

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

The default byte-level tokenizer can encode prompts containing characters that were not seen in the training corpus.

## Inspect Tokenizer

```bash
python scripts/inspect_tokenizer.py --dataset data/input.txt
```

The inspector uses the subword tokenizer by default. To inspect the character tokenizer instead:

```bash
python scripts/inspect_tokenizer.py --dataset data/input.txt --tokenizer char
```

To inspect or train a SentencePiece tokenizer:

```bash
python scripts/inspect_tokenizer.py --dataset data/input.txt --tokenizer sentencepiece --tokenizer-prefix tokenizers/sentencepiece/local_4096 --vocab-size 4096
```

For a URL list:

```bash
python scripts/inspect_tokenizer.py --dataset data/my_links_corpus.txt --url-list --cache data/fetched_links_corpus.txt
```

For Wikipedia page titles:

```bash
python scripts/inspect_tokenizer.py --dataset data/my_wikipedia_pages.txt --wikipedia-titles --cache data/fetched_wikipedia_corpus.txt
```

To inspect the tokenizer stored in a checkpoint:

```bash
python scripts/inspect_checkpoint_tokenizer.py --checkpoint checkpoints/latest.pt --sample "A tortilla is a traditional Mexican flatbread."
```

## Tests

```bash
pytest
```

## Project Layout

```text
toy_llm/
  config.py       YAML-backed dataclass config
  tokenizer.py    character tokenizer and byte-level BPE tokenizer
  dataset.py      UTF-8 loading and random batch sampling
  model.py        tiny GPT-style decoder-only Transformer
  train_loop.py   evaluation, training, checkpoint scheduling
  checkpoint.py   save/load helpers
  device.py       auto CUDA/MPS/CPU selection
  sampling.py     small model utilities
```
