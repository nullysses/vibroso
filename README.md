<p align="center">
  <img src="assets/vibroso-logo.png" alt="Vibroso logo" width="720">
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

## Quick Start

Train a tiny local model:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Generate from the latest checkpoint:

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --max-new-tokens 300
```

Fine-tune an instruction dataset from a base checkpoint:

```bash
python scripts/train.py --config configs/instruction_seed.yaml --init-from checkpoints/base/latest.pt
```

## Train

Put a UTF-8 text corpus at `data/input.txt`, then run:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Checkpoints are written to `checkpoints/latest.pt`.

The trainer prints startup timing for loading the corpus, building or loading the tokenizer, encoding the dataset, initializing the model, and printing startup stats. Training metrics can also be written to JSONL:

```bash
python scripts/train.py --config configs/tiny.yaml --metrics runs/tiny_metrics.jsonl
```

On CUDA devices, training uses bf16 mixed precision with PyTorch autocast and GradScaler. CPU and MPS training leave autocast disabled.

### Tokenizers

The default tokenizer is a small trainable byte-level BPE tokenizer. It starts with all 256 byte values, learns frequent adjacent byte-token merges from the training corpus, and stores the learned merges in the checkpoint.

Supported `tokenizer_kind` values:

```yaml
tokenizer_kind: subword       # default byte-level BPE
tokenizer_kind: char          # original character-level tokenizer
tokenizer_kind: sentencepiece # SentencePiece backend
```

After tokenizer changes, retrain from scratch instead of resuming older subword checkpoints. Old subword tokenizer payloads are rejected so model weights are not reused with incompatible token IDs.

For large corpora, `tokenizer_train_chars` controls how many leading corpus characters are used to learn BPE merges. The base byte vocabulary still covers any valid UTF-8 text. The Wikipedia config starts with `tokenizer_vocab_size: 2048` and `tokenizer_train_chars: 10000`; increase the training window only after confirming startup time is acceptable.

SentencePiece example:

```bash
python scripts/train.py --config configs/sentencepiece.yaml
```

### Model Options

The Transformer uses RoPE positional encoding, RMSNorm blocks, batched causal self-attention, and KV-cache generation. The MLP is configurable with `model.mlp_type: gelu` or `model.mlp_type: swiglu`. GELU remains the default for old configs and checkpoints. SwiGLU changes model weights and requires training a new checkpoint with `mlp_type: swiglu`.

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

Wikipedia training has three startup phases before model steps begin: fetch/cache pages, build the tokenizer, then encode the cached corpus. The trainer prints `building tokenizer...` and `encoding dataset...` so you can tell which phase is running.

### Dataset Sources

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

### Resume vs Fine-Tune

To resume training:

```bash
python scripts/train.py --config configs/tiny.yaml --resume checkpoints/latest.pt
```

To fine-tune from a base checkpoint on a new dataset, use `--init-from`. This loads the checkpoint model weights and tokenizer, but keeps the supplied config, starts at step 0, uses a fresh optimizer, and writes to the supplied `checkpoint_dir`.

```bash
python scripts/train.py --config configs/instruction_tiny.yaml --init-from checkpoints/base/latest.pt
```

`--init-from` requires the new config to match the checkpoint `block_size` and model architecture.

## Instruction JSONL

Small instruction datasets can be loaded with `dataset_kind: instruction_jsonl`. Each line must be a JSON object with `instruction` and `response`; `context` is optional.

```jsonl
{"instruction":"What is a tortilla?","response":"A tortilla is a thin flatbread made from corn or wheat flour."}
```

Preview the formatted chat-style corpus:

```bash
python scripts/preview_instruction_dataset.py --dataset data/instruction_examples.example.jsonl --limit 3
```

Train the tiny example:

```bash
python scripts/train.py --config configs/instruction_tiny.yaml
```

Fine-tune from a base checkpoint with the larger example config:

```bash
python scripts/train.py --config configs/instruction_seed.yaml --init-from checkpoints/base/latest.pt
```

Instruction prompts should use the same markers:

```text
<|user|>
What is a tortilla?
<|assistant|>
```

## Generate

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --max-new-tokens 300
```

Generation options:

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --temperature 0.8 --top-k 40
```

For instruction-tuned checkpoints, use `--stop` to trim generation at a marker such as `<|end|>`. Pass `--keep-stop` if the marker should remain in the printed text.

```bash
python scripts/generate.py --checkpoint checkpoints/instruct/latest.pt --prompt "<|user|>\nWhat is a tortilla?\n<|assistant|>\n" --temperature 0.4 --top-k 20 --max-new-tokens 120 --stop "<|end|>"
```

Generation can also use a YAML config:

```bash
python scripts/generate.py --config configs/inference.yaml
```

Inference config example:

```yaml
checkpoint: checkpoints/instruct_seed123/latest.pt
prompt: |
  <|user|>
  What is a tortilla?
  <|assistant|>
max_new_tokens: 120
temperature: 0.35
top_k: 20
stop:
  - "<|end|>"
keep_stop: false
device: auto
seed: 1337
```

The default byte-level tokenizer can encode prompts containing characters that were not seen in the training corpus.

## Benchmarks

Benchmark tokenizer compression for a training config:

```bash
python scripts/benchmark_tokenizer.py --config configs/wikipedia.yaml --output runs/tokenizer_benchmark.json
```

Benchmark generation samples from a checkpoint:

```bash
python scripts/benchmark_generation.py --checkpoint checkpoints/latest.pt --max-new-tokens 100 --output runs/generation_benchmark.txt
```

`scripts/benchmark_generation.py` uses built-in prompts unless `benchmarks/prompts.txt` exists or `--prompt-file` is provided. Separate prompt blocks with `---`.

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

## Config Reference

Common training fields:

```yaml
dataset_path: data/input.txt
dataset_kind: text              # text, url_list, wikipedia_titles, instruction_jsonl
tokenizer_kind: subword         # subword, char, sentencepiece
checkpoint_dir: checkpoints
device: auto                    # auto, cuda, mps, cpu
batch_size: 16
block_size: 64
max_steps: 2000
learning_rate: 0.001
weight_decay: 0.1

model:
  n_embd: 128
  n_head: 4
  n_layer: 2
  dropout: 0.1
  mlp_type: gelu                # gelu, swiglu
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
