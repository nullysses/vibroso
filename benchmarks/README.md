# Vibroso Benchmarks

This directory stores benchmark prompts and generated benchmark outputs.

Benchmarking currently focuses on:

1. tokenizer compression
2. validation loss / perplexity
3. fixed-prompt generation quality
4. training and generation speed

## Recommended Benchmark Flow

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

To record training metrics:

```powershell
python scripts/train.py --config configs/links.yaml --metrics benchmarks/runs/latest/metrics.jsonl
```

## Interpretation

Raw loss should only be compared across compatible tokenizers and datasets.

Generation samples should be compared using the same prompts, seed, temperature, top-k, and max token settings.
