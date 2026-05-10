# Change Request: Add Benchmark Scripts for Tokenizer, Generation, and Training Metrics

## Summary

Add a small benchmarking suite to Vibroso so model changes can be evaluated consistently instead of relying only on ad hoc generation samples.

The benchmark suite should cover:

1. **Tokenizer compression and sample tokenization**
2. **Fixed-prompt generation quality**
3. **Training/runtime metrics**
4. **Run artifact organization**

This will make it easier to compare changes such as byte-level BPE, RoPE, RMSNorm, SwiGLU, corpus cleaning, and model-size changes.

---

## Goals

### Primary goals

- Make Vibroso runs comparable across model/tokenizer/config changes.
- Track whether tokenizer changes improve compression.
- Generate fixed prompt samples with deterministic seeds.
- Record core training metrics in a machine-readable format.
- Provide a standard place for benchmark outputs.

### Non-goals

- Do not implement full academic benchmark suites.
- Do not benchmark instruction-following yet.
- Do not require external datasets beyond the configured corpus.
- Do not introduce heavy dependencies.

---

## Required Files

Add:

```text
scripts/benchmark_tokenizer.py
scripts/benchmark_generation.py
benchmarks/README.md
benchmarks/prompts.txt
benchmarks/runs/.gitkeep
```

Optional but useful:

```text
scripts/benchmark_training.py
```

---

## 1. Tokenizer Benchmark Script

### File

```text
scripts/benchmark_tokenizer.py
```

### Purpose

Benchmark tokenizer quality and compression for a given training config.

### CLI

```powershell
python scripts/benchmark_tokenizer.py --config configs/links.yaml
```

### Behavior

The script should:

1. Load `TrainConfig`.
2. Load corpus via existing `load_corpus(config)`.
3. Build tokenizer using existing `build_tokenizer(...)`.
4. Encode the full corpus.
5. Print tokenizer statistics.
6. Print sample tokenizations for fixed sample strings.

### Required output

Print:

```text
=== Tokenizer Benchmark ===
tokenizer kind: subword
vocab size: 4096
merge count: 3840
dataset chars: 1,234,567
dataset tokens: 345,678
chars/token: 3.571
```

If available, also print:

```text
bytes/token
```

### Sample strings

Use these default samples:

```python
SAMPLES = [
    "A tortilla is a traditional Mexican flatbread.",
    "Artificial intelligence is a field of computer science.",
    "The Roman Empire was one of the largest empires in history.",
    "Machine learning models are trained using data.",
]
```

For each sample, print:

```text
sample: 'A tortilla is a traditional Mexican flatbread.'
tokens: 12
chars/token: 3.833
pieces:
['A', ' tortilla', ' is', ' a', ' traditional', ' Mexican', ' flat', 'bread', '.']
```

The exact pieces will vary by tokenizer.

### Acceptance criteria

- Works with both `char` and `subword` tokenizers.
- Reports corpus-level token count.
- Reports `chars/token`.
- Uses existing project config/data/tokenizer code.
- Does not train a model.
- Does not write files unless an optional output path is supplied.

---

## 2. Generation Benchmark Script

### File

```text
scripts/benchmark_generation.py
```

### Purpose

Generate samples from a fixed prompt suite using deterministic settings so different checkpoints can be compared.

### CLI

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.3 `
  --top-k 10 `
  --max-new-tokens 100 `
  --seed 1337
```

Optional output:

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.3 `
  --top-k 10 `
  --max-new-tokens 100 `
  --seed 1337 `
  --output benchmarks/runs/run_001/generation_cold.txt
```

### Default prompts

Use these prompts:

```python
PROMPTS = [
    "Tortilla\nA tortilla is",
    "Artificial intelligence\nArtificial intelligence is",
    "Physics\nPhysics is",
    "Music\nMusic is",
    "Mexico\nMexico is",
    "The Roman Empire\nThe Roman Empire was",
]
```

Also support loading prompts from:

```text
benchmarks/prompts.txt
```

Prompt file format:

```text
---
Tortilla
A tortilla is
---
Artificial intelligence
Artificial intelligence is
---
Physics
Physics is
```

Each prompt is separated by a line containing only:

```text
---
```

### Behavior

The script should:

1. Load checkpoint.
2. Restore `TrainConfig` from checkpoint config.
3. Restore tokenizer from checkpoint.
4. Reconstruct `TinyGPT`.
5. Load model weights.
6. Set model to eval mode.
7. Set random seed if provided.
8. Generate text for each prompt.
9. Print or save results.

### Required output format

```text
=== Generation Benchmark ===
checkpoint: checkpoints/latest.pt
temperature: 0.3
top_k: 10
max_new_tokens: 100
seed: 1337

================================================================================
PROMPT: 'Tortilla\nA tortilla is'
--------------------------------------------------------------------------------
Tortilla
A tortilla is ...

================================================================================
PROMPT: 'Artificial intelligence\nArtificial intelligence is'
--------------------------------------------------------------------------------
Artificial intelligence
Artificial intelligence is ...
```

### Acceptance criteria

- Uses the tokenizer stored in the checkpoint.
- Uses the training config stored in the checkpoint.
- Supports `--prompt-file`.
- Supports `--output`.
- Supports `--seed`.
- Produces deterministic output when using the same seed/device/settings, within normal PyTorch limitations.
- Does not modify checkpoint files.

---

## 3. Training Metrics Logging

### Purpose

Record run metrics in a standard format so changes can be compared.

### Required change

During training, optionally write a JSON metrics file.

Add config field or CLI option:

```yaml
metrics_path: benchmarks/runs/latest/metrics.json
```

or CLI:

```powershell
python scripts/train.py --config configs/links.yaml --metrics benchmarks/runs/run_001/metrics.json
```

Either approach is acceptable.

### Metrics to record

At minimum:

```json
{
  "run_name": "bpe4096_rope_rmsnorm_256x4",
  "vocab_size": 4096,
  "tokenizer_kind": "subword",
  "tokenizer_train_chars": 500000,
  "dataset_chars": 1234567,
  "dataset_tokens": 345678,
  "chars_per_token": 3.57,
  "parameters": 12345678,
  "block_size": 256,
  "batch_size": 32,
  "max_steps": 20000,
  "learning_rate": 0.0003,
  "weight_decay": 0.1,
  "model": {
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1
  },
  "final_train_loss": 1.82,
  "final_val_loss": 1.96,
  "final_val_perplexity": 7.10,
  "tokens_per_sec": 21000
}
```

If final losses are not easily available from the current train loop, first implement periodic append-style JSONL logging instead:

```jsonl
{"step":0,"train_loss":8.32,"val_loss":8.29,"val_perplexity":3980.2,"tokens_per_sec":0}
{"step":500,"train_loss":4.12,"val_loss":4.21,"val_perplexity":67.36,"tokens_per_sec":21000}
{"step":1000,"train_loss":3.75,"val_loss":3.92,"val_perplexity":50.40,"tokens_per_sec":21250}
```

JSONL is acceptable and may be simpler.

### Acceptance criteria

- Training can run without metrics logging.
- If metrics path is supplied, parent directories are created automatically.
- Metrics include at least step, train loss, val loss, val perplexity, and tokens/sec.
- Metrics format is valid JSON or JSONL.
- Existing training behavior remains unchanged when no metrics path is supplied.

---

## 4. Benchmarks Directory

Add:

```text
benchmarks/
  README.md
  prompts.txt
  runs/
    .gitkeep
```

### `benchmarks/README.md` contents

Include:

````md
# Vibroso Benchmarks

This directory stores benchmark prompts and generated benchmark outputs.

Benchmarking currently focuses on:

1. tokenizer compression
2. validation loss / perplexity
3. fixed-prompt generation quality
4. training and generation speed

## Recommended benchmark flow

```powershell
python scripts/benchmark_tokenizer.py --config configs/links.yaml

python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.3 `
  --top-k 10 `
  --max-new-tokens 100 `
  --seed 1337 `
  --output benchmarks/runs/latest/generation_cold.txt

python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.7 `
  --top-k 40 `
  --max-new-tokens 150 `
  --seed 1337 `
  --output benchmarks/runs/latest/generation_warm.txt
```

## Interpretation

Raw loss should only be compared across compatible tokenizers and datasets.

Generation samples should be compared using the same prompts, seed, temperature, top-k, and max token settings.
````

### `benchmarks/prompts.txt`

Add:

```text
---
Tortilla
A tortilla is
---
Artificial intelligence
Artificial intelligence is
---
Physics
Physics is
---
Music
Music is
---
Mexico
Mexico is
---
The Roman Empire
The Roman Empire was
```

---

## 5. Suggested Implementation Sketch

### `scripts/benchmark_tokenizer.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.config import TrainConfig
from toy_llm.dataset import load_corpus
from toy_llm.tokenizer import build_tokenizer


SAMPLES = [
    "A tortilla is a traditional Mexican flatbread.",
    "Artificial intelligence is a field of computer science.",
    "The Roman Empire was one of the largest empires in history.",
    "Machine learning models are trained using data.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Vibroso tokenizer compression.")
    parser.add_argument("--config", default="configs/links.yaml")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    text = load_corpus(config)

    tokenizer = build_tokenizer(
        text,
        kind=config.tokenizer_kind,
        vocab_size=config.tokenizer_vocab_size,
        max_train_chars=config.tokenizer_train_chars,
    )

    ids = tokenizer.encode(text)

    print("=== Tokenizer Benchmark ===")
    print(f"tokenizer kind: {config.tokenizer_kind}")
    print(f"vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "merge_count"):
        print(f"merge count: {tokenizer.merge_count}")
    print(f"dataset chars: {len(text):,}")
    print(f"dataset tokens: {len(ids):,}")
    print(f"chars/token: {len(text) / max(len(ids), 1):.3f}")

    print()
    print("=== Sample Encodings ===")
    for sample in SAMPLES:
        sample_ids = tokenizer.encode(sample)
        print()
        print(f"sample: {sample!r}")
        print(f"tokens: {len(sample_ids)}")
        print(f"chars/token: {len(sample) / max(len(sample_ids), 1):.3f}")

        if hasattr(tokenizer, "encode_to_pieces"):
            print("pieces:")
            print(tokenizer.encode_to_pieces(sample))


if __name__ == "__main__":
    main()
```

---

### `scripts/benchmark_generation.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.config import TrainConfig
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import tokenizer_from_dict


DEFAULT_PROMPTS = [
    "Tortilla\nA tortilla is",
    "Artificial intelligence\nArtificial intelligence is",
    "Physics\nPhysics is",
    "Music\nMusic is",
    "Mexico\nMexico is",
    "The Roman Empire\nThe Roman Empire was",
]


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS

    text = Path(path).read_text(encoding="utf-8")
    prompts = [
        block.strip("\n")
        for block in text.split("---")
        if block.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Vibroso generation.")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    train_config = TrainConfig.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=train_config.block_size,
        n_embd=train_config.model.n_embd,
        n_head=train_config.model.n_head,
        n_layer=train_config.model.n_layer,
        dropout=train_config.model.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompts = load_prompts(args.prompt_file)

    sections: list[str] = []
    sections.append("=== Generation Benchmark ===")
    sections.append(f"checkpoint: {args.checkpoint}")
    sections.append(f"temperature: {args.temperature}")
    sections.append(f"top_k: {args.top_k}")
    sections.append(f"max_new_tokens: {args.max_new_tokens}")
    sections.append(f"seed: {args.seed}")
    sections.append("")

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        text = tokenizer.decode(out[0].tolist())

        sections.append("=" * 80)
        sections.append(f"PROMPT: {prompt!r}")
        sections.append("-" * 80)
        sections.append(text)
        sections.append("")

    output_text = "\n".join(sections)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
```

---

## 6. Validation Commands

After implementation, these commands should work:

```powershell
python scripts/benchmark_tokenizer.py --config configs/links.yaml
```

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.3 `
  --top-k 10 `
  --max-new-tokens 100 `
  --seed 1337
```

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --prompt-file benchmarks/prompts.txt `
  --temperature 0.7 `
  --top-k 40 `
  --max-new-tokens 150 `
  --seed 1337 `
  --output benchmarks/runs/latest/generation_warm.txt
```

---

## Definition of Done

This change is complete when:

- `benchmark_tokenizer.py` runs successfully against an existing config.
- `benchmark_generation.py` runs successfully against an existing checkpoint.
- Prompt files are supported.
- Output files are supported.
- `benchmarks/README.md` documents the workflow.
- Benchmark outputs are reproducible with fixed seed/settings.
- No existing training or generation commands are broken.
