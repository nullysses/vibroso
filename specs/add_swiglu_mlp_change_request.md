# Change Request: Replace Plain GELU MLP with Optional SwiGLU MLP

## Summary

Add **SwiGLU** as an optional feed-forward/MLP implementation in Vibroso's Transformer blocks.

The current model already has several modern LLM components:

- SentencePiece tokenizer support
- RoPE positional encoding
- RMSNorm
- fused QKV attention
- KV-cache generation

However, the Transformer block still uses a classic plain MLP:

```text
Linear(n_embd -> 4 * n_embd)
GELU
Linear(4 * n_embd -> n_embd)
Dropout
```

This change adds a modern gated MLP alternative:

```text
gate_proj(x), up_proj(x)
SiLU(gate_proj(x)) * up_proj(x)
down_proj(...)
Dropout
```

This should be configurable so Vibroso can benchmark:

```text
RoPE + RMSNorm + GELU MLP
vs
RoPE + RMSNorm + SwiGLU MLP
```

---

## Current State

In `toy_llm/model.py`, the current feed-forward block is:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

And `Block` currently wires it directly:

```python
self.ffwd = FeedForward(n_embd, dropout)
```

This change should keep that implementation as the baseline while adding `SwiGLUFeedForward`.

---

## Goals

### Primary goals

- Add a `SwiGLUFeedForward` module.
- Add a config field to choose MLP type.
- Preserve the current GELU MLP as a baseline.
- Make the selected MLP type part of the checkpoint config.
- Add tests for shape correctness and config validation.
- Ensure old behavior remains available with `mlp_type: gelu`.

### Non-goals

- Do not remove the existing `FeedForward` implementation.
- Do not change the tokenizer.
- Do not change RoPE.
- Do not change RMSNorm.
- Do not change KV-cache behavior.
- Do not make old checkpoints magically compatible with SwiGLU.
- Do not implement Mixture-of-Experts or other MLP variants.

---

## Proposed Config Change

Update `ModelConfig` in `toy_llm/config.py` to include:

```python
mlp_type: str = "gelu"
```

Allowed values:

```text
gelu
swiglu
```

Example config:

```yaml
model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.1
  mlp_type: swiglu
```

Validation rule:

```python
if self.mlp_type not in {"gelu", "swiglu"}:
    raise ValueError("model.mlp_type must be 'gelu' or 'swiglu'")
```

---

## New Module: `SwiGLUFeedForward`

Add this class in `toy_llm/model.py` near the existing `FeedForward` class.

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        n_embd: int,
        dropout: float,
        multiple_of: int = 8,
    ):
        super().__init__()

        hidden_dim = int((8 / 3) * n_embd)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)
```

---

## Why `8 / 3 * n_embd`?

The current GELU MLP has approximately:

```text
n_embd -> 4 * n_embd -> n_embd
```

Parameter count, ignoring bias:

```text
n_embd * 4n_embd + 4n_embd * n_embd
= 8 * n_embd^2
```

SwiGLU has three projections:

```text
gate_proj: n_embd -> hidden_dim
up_proj:   n_embd -> hidden_dim
down_proj: hidden_dim -> n_embd
```

Parameter count:

```text
3 * n_embd * hidden_dim
```

To keep parameter count close to the current MLP:

```text
3 * n_embd * hidden_dim ~= 8 * n_embd^2
hidden_dim ~= 8/3 * n_embd
```

So for:

```yaml
n_embd: 256
```

SwiGLU uses roughly:

```text
hidden_dim ~= 682
```

rounded up to a clean multiple, e.g.:

```text
688
```

This avoids making SwiGLU much larger than the current 4x GELU MLP.

---

## Update `Block`

Change the `Block` constructor to accept `mlp_type`.

Current:

```python
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
```

Proposed:

```python
class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        mlp_type: str = "gelu",
    ):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)

        if mlp_type == "gelu":
            self.ffwd = FeedForward(n_embd, dropout)
        elif mlp_type == "swiglu":
            self.ffwd = SwiGLUFeedForward(n_embd, dropout)
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type!r}")

        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
```

The `forward` and `forward_with_cache` methods should not need to change, because both MLPs preserve shape:

```text
[B, T, n_embd] -> [B, T, n_embd]
```

---

## Update `TinyGPT`

Update the `TinyGPT` constructor to accept `mlp_type`.

Current:

```python
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
```

Proposed:

```python
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        mlp_type: str = "gelu",
    ):
```

And when constructing blocks:

```python
self.blocks = nn.ModuleList(
    [
        Block(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            mlp_type=mlp_type,
        )
        for _ in range(n_layer)
    ]
)
```

---

## Update Training and Generation Scripts

Wherever `TinyGPT` is instantiated, pass:

```python
mlp_type=train_config.model.mlp_type
```

Affected scripts likely include:

```text
scripts/train.py
scripts/generate.py
scripts/benchmark_generation.py
```

Any tests or utilities that instantiate `TinyGPT` directly should either pass `mlp_type` or rely on the default `"gelu"`.

---

## Checkpoint Compatibility

This is a checkpoint-breaking architecture change **when `mlp_type: swiglu` is used**.

Existing GELU checkpoints have keys like:

```text
blocks.0.ffwd.net.0.weight
blocks.0.ffwd.net.0.bias
blocks.0.ffwd.net.2.weight
blocks.0.ffwd.net.2.bias
```

SwiGLU checkpoints will have keys like:

```text
blocks.0.ffwd.gate_proj.weight
blocks.0.ffwd.up_proj.weight
blocks.0.ffwd.down_proj.weight
```

Therefore:

- Old GELU checkpoints should still load when config uses `mlp_type: gelu`.
- New SwiGLU checkpoints require `mlp_type: swiglu`.
- Do not expect GELU checkpoints to load into a SwiGLU model.
- Do not expect SwiGLU checkpoints to load into a GELU model.

The checkpoint config must preserve `model.mlp_type`.

---

## Suggested Example Config

Add or update a config such as:

```text
configs/sentencepiece_swiglu.yaml
```

Example:

```yaml
dataset_path: data/my_wikipedia_pages.txt
dataset_kind: wikipedia_titles

tokenizer_kind: sentencepiece
tokenizer_vocab_size: 4096
tokenizer_model_type: bpe
tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model
tokenizer_train_chars: 1000000

corpus_cache_path: data/fetched_wikipedia_corpus.txt
wikipedia_language: en
user_agent: "vibroso/0.1 academic corpus builder"

checkpoint_dir: checkpoints

seed: 1337
device: cuda

train_split: 0.9
batch_size: 16
block_size: 256

max_steps: 10000
eval_interval: 500
eval_iters: 50
checkpoint_interval: 1000
learning_rate: 0.0003
weight_decay: 0.1

model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.1
  mlp_type: swiglu
```

---

## Tests

Add or update tests to cover both MLP implementations.

### 1. Feed-forward shape test

```python
def test_swiglu_feedforward_preserves_shape():
    ff = SwiGLUFeedForward(n_embd=256, dropout=0.1)
    x = torch.randn(4, 32, 256)
    y = ff(x)
    assert y.shape == x.shape
```

### 2. TinyGPT forward test with GELU

```python
def test_tinygpt_forward_with_gelu_mlp():
    model = TinyGPT(
        vocab_size=4096,
        block_size=128,
        n_embd=256,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        mlp_type="gelu",
    )
    idx = torch.randint(0, 4096, (2, 64))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 64, 4096)
    assert loss is not None
```

### 3. TinyGPT forward test with SwiGLU

```python
def test_tinygpt_forward_with_swiglu_mlp():
    model = TinyGPT(
        vocab_size=4096,
        block_size=128,
        n_embd=256,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        mlp_type="swiglu",
    )
    idx = torch.randint(0, 4096, (2, 64))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 64, 4096)
    assert loss is not None
```

### 4. Config validation test

```python
def test_model_config_rejects_unknown_mlp_type():
    with pytest.raises(ValueError):
        ModelConfig(mlp_type="unknown").validate()
```

---

## Benchmarking Guidance

Benchmark SwiGLU against the current GELU MLP with all other variables held constant:

```text
same tokenizer
same corpus
same model size
same block_size
same seed
same batch_size
same learning rate
same number of steps
same benchmark prompts
```

Suggested comparison:

```text
Run A: sentencepiece + RoPE + RMSNorm + GELU
Run B: sentencepiece + RoPE + RMSNorm + SwiGLU
```

Track:

- final train loss
- final validation loss
- validation perplexity
- tokens/sec
- generation benchmark samples
- repetition behavior
- checkpoint size
- parameter count

Do not judge the change by a single generation sample.

---

## Acceptance Criteria

This change is complete when:

1. `ModelConfig` supports `mlp_type`.
2. Valid values are `gelu` and `swiglu`.
3. Invalid values raise a clear `ValueError`.
4. Existing `FeedForward` remains available as the GELU baseline.
5. New `SwiGLUFeedForward` exists and preserves input shape.
6. `Block` selects the correct MLP based on `mlp_type`.
7. `TinyGPT` accepts and passes `mlp_type`.
8. Training script passes `train_config.model.mlp_type` into `TinyGPT`.
9. Generation script passes checkpoint config `model.mlp_type` into `TinyGPT`.
10. Benchmark generation script, if present, passes `model.mlp_type` into `TinyGPT`.
11. Existing configs without `mlp_type` keep working with default `gelu`.
12. A new config using `mlp_type: swiglu` trains from scratch successfully.
13. Tests cover GELU and SwiGLU forward paths.
14. Documentation or README notes that `mlp_type: swiglu` is checkpoint-incompatible with old GELU checkpoints.

---

## Validation Commands

Run tests:

```powershell
pytest
```

Train a SwiGLU model from scratch:

```powershell
python scripts/train.py --config configs/sentencepiece_swiglu.yaml
```

Generate from the resulting checkpoint:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/latest.pt `
  --prompt "Tortilla`nA tortilla is" `
  --temperature 0.5 `
  --top-k 30 `
  --max-new-tokens 120
```

Run benchmark generation, if benchmark scripts exist:

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints/latest.pt `
  --temperature 0.3 `
  --top-k 10 `
  --max-new-tokens 100 `
  --seed 1337 `
  --output benchmarks/runs/latest/generation_cold.txt
```

---

## Notes

SwiGLU is a worthwhile modernization, but it is not expected to fix poor corpus quality, undertraining, or tokenizer issues by itself.

Expected improvement should be evaluated through validation loss, fixed-prompt benchmark samples, and repetition behavior after training from scratch.
