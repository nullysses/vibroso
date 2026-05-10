# Change Request: Add SentencePiece as an Optional Tokenizer Backend

## Summary

Add **SentencePiece** as a supported tokenizer backend in Vibroso, alongside the existing `char` and custom `subword` tokenizers.

The current custom byte-BPE tokenizer is useful educationally, but it is implemented in pure Python and becomes slow on large corpora. SentencePiece provides a mature, optimized tokenizer implementation that can train and encode large text corpora much faster, while preserving Vibroso’s learning-oriented architecture.

---

## Goals

### Primary goals

- Add `sentencepiece` as a valid `tokenizer_kind`.
- Support training a SentencePiece model from the configured corpus.
- Support encoding/decoding through SentencePiece.
- Store enough tokenizer metadata in checkpoints to reload the tokenizer during generation.
- Keep existing `char` and custom `subword` tokenizers working.
- Make large-corpus tokenization much faster.

### Non-goals

- Do not remove the custom tokenizer.
- Do not require SentencePiece for all runs.
- Do not change the model architecture.
- Do not implement instruction tuning as part of this change.
- Do not make SentencePiece checkpoints compatible with previous custom-tokenizer checkpoints.

---

## Required Dependency

Add to `requirements.txt`:

```text
sentencepiece
```

Optional import behavior is acceptable, but if `tokenizer_kind: sentencepiece` is selected and the package is missing, the program should fail with a clear error:

```text
SentencePiece tokenizer requested, but sentencepiece is not installed.
Install it with: pip install sentencepiece
```

---

## Config Changes

Update `TrainConfig` to allow:

```yaml
tokenizer_kind: sentencepiece
```

Current accepted values:

```text
char
subword
```

New accepted values:

```text
char
subword
sentencepiece
```

Add optional config fields:

```yaml
tokenizer_kind: sentencepiece
tokenizer_vocab_size: 4096
tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model
tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
tokenizer_model_type: bpe
```

Recommended valid `tokenizer_model_type` values:

```text
bpe
unigram
```

Default:

```yaml
tokenizer_model_type: bpe
```

Suggested example config:

```yaml
dataset_path: data/my_wikipedia_pages.txt
dataset_kind: wikipedia_titles

tokenizer_kind: sentencepiece
tokenizer_vocab_size: 4096
tokenizer_model_type: bpe
tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model

corpus_cache_path: data/fetched_wikipedia_corpus.txt
wikipedia_language: en
user_agent: "vibroso/0.1 academic corpus builder"

checkpoint_dir: checkpoints
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
```

---

## New Tokenizer Class

Add a new class in `toy_llm/tokenizer.py`:

```python
class SentencePieceTokenizer:
    ...
```

### Required interface

It must satisfy the existing `Tokenizer` protocol:

```python
@property
def vocab_size(self) -> int:
    ...

def encode(self, text: str) -> list[int]:
    ...

def decode(self, ids: list[int] | tuple[int, ...]) -> str:
    ...

def to_dict(self) -> dict:
    ...
```

### Suggested constructor

```python
class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        ...
```

### Suggested implementation sketch

```python
class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencePiece tokenizer requested, but sentencepiece is not installed. "
                "Install it with: pip install sentencepiece"
            ) from exc

        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor()
        loaded = self.processor.Load(self.model_path)
        if not loaded:
            raise ValueError(f"Failed to load SentencePiece model: {self.model_path}")

    @property
    def vocab_size(self) -> int:
        return int(self.processor.GetPieceSize())

    def encode(self, text: str) -> list[int]:
        return list(self.processor.EncodeAsIds(text))

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.processor.DecodeIds([int(i) for i in ids])

    def encode_to_pieces(self, text: str) -> list[str]:
        return list(self.processor.EncodeAsPieces(text))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "sentencepiece",
            "algorithm": "sentencepiece",
            "version": 1,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SentencePieceTokenizer":
        model_path = data.get("model_path")
        if not model_path:
            raise ValueError("SentencePiece tokenizer checkpoint is missing 'model_path'")
        return cls(str(model_path))
```

---

## Training SentencePiece

Add helper function:

```python
def train_sentencepiece_tokenizer(
    text: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
) -> SentencePieceTokenizer:
    ...
```

### Suggested behavior

1. Write corpus text to a temporary training file.
2. Call `sentencepiece.SentencePieceTrainer.Train(...)`.
3. Save model files to:

```text
{model_prefix}.model
{model_prefix}.vocab
```

4. Return `SentencePieceTokenizer(f"{model_prefix}.model")`.

### Suggested implementation sketch

```python
def train_sentencepiece_tokenizer(
    text: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
) -> SentencePieceTokenizer:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError(
            "SentencePiece tokenizer requested, but sentencepiece is not installed. "
            "Install it with: pip install sentencepiece"
        ) from exc

    if model_type not in {"bpe", "unigram"}:
        raise ValueError("tokenizer_model_type must be 'bpe' or 'unigram'")

    prefix_path = Path(model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    training_path = prefix_path.with_suffix(".train.txt")
    training_path.write_text(text, encoding="utf-8")

    spm.SentencePieceTrainer.Train(
        input=str(training_path),
        model_prefix=str(prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )

    return SentencePieceTokenizer(str(prefix_path.with_suffix(".model")))
```

---

## `build_tokenizer` Changes

Update `build_tokenizer(...)` to support:

```python
kind == "sentencepiece"
```

Suggested signature change:

```python
def build_tokenizer(
    text: str,
    kind: str = "subword",
    vocab_size: int = 256,
    max_train_chars: int | None = 10_000,
    sentencepiece_model_path: str | None = None,
    sentencepiece_prefix: str | None = None,
    sentencepiece_model_type: str = "bpe",
) -> Tokenizer:
    ...
```

Expected behavior:

```python
if kind == "sentencepiece":
    if sentencepiece_model_path and Path(sentencepiece_model_path).exists():
        return SentencePieceTokenizer(sentencepiece_model_path)

    if not sentencepiece_prefix:
        raise ValueError(
            "tokenizer_prefix must be provided when training a SentencePiece tokenizer"
        )

    train_text = text if max_train_chars is None else text[:max_train_chars]
    return train_sentencepiece_tokenizer(
        train_text,
        model_prefix=sentencepiece_prefix,
        vocab_size=vocab_size,
        model_type=sentencepiece_model_type,
    )
```

---

## Checkpoint Serialization

### Required

When training with SentencePiece, the checkpoint tokenizer payload should include:

```json
{
  "kind": "sentencepiece",
  "algorithm": "sentencepiece",
  "version": 1,
  "model_path": "tokenizers/sentencepiece/wiki_4096.model"
}
```

### Important

The checkpoint should not assume that the SentencePiece model file is globally available unless the path exists relative to the project root.

Generation should fail clearly if the model file is missing:

```text
SentencePiece model file not found: tokenizers/sentencepiece/wiki_4096.model
This checkpoint requires the tokenizer model used during training.
```

---

## `tokenizer_from_dict` Changes

Update:

```python
def tokenizer_from_dict(data: dict[str, object]) -> Tokenizer:
    ...
```

Add:

```python
if kind == "sentencepiece":
    return SentencePieceTokenizer.from_dict(data)
```

Existing behavior for `char` and `subword` should remain unchanged.

---

## Training Script Changes

Update `scripts/train.py` so that when it calls `build_tokenizer(...)`, it passes the SentencePiece-specific config fields.

Current style:

```python
tokenizer = build_tokenizer(
    text,
    kind=config.tokenizer_kind,
    vocab_size=config.tokenizer_vocab_size,
    max_train_chars=config.tokenizer_train_chars,
)
```

Update to something like:

```python
tokenizer = build_tokenizer(
    text,
    kind=config.tokenizer_kind,
    vocab_size=config.tokenizer_vocab_size,
    max_train_chars=config.tokenizer_train_chars,
    sentencepiece_model_path=config.tokenizer_model_path,
    sentencepiece_prefix=config.tokenizer_prefix,
    sentencepiece_model_type=config.tokenizer_model_type,
)
```

---

## Generation Script Changes

No major changes should be needed if:

1. `tokenizer_from_dict()` supports SentencePiece.
2. The checkpoint contains the SentencePiece model path.
3. The model file exists locally.

Existing generation flow should continue working:

```python
checkpoint = load_checkpoint(...)
tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
prompt_ids = tokenizer.encode(prompt)
...
print(tokenizer.decode(...))
```

---

## Benchmark Script Changes

If benchmark scripts exist, update them to support SentencePiece automatically through the standard tokenizer interface.

Tokenizer benchmark should print pieces if available:

```python
if hasattr(tokenizer, "encode_to_pieces"):
    print(tokenizer.encode_to_pieces(sample))
```

Expected SentencePiece output may look like:

```text
['▁A', '▁tortilla', '▁is', '▁a', '▁traditional', '▁Mexican', '▁flat', 'bread', '.']
```

---

## Suggested `.gitignore` Updates

SentencePiece creates tokenizer model artifacts. Decide whether to commit them.

For reproducibility, it may be useful to commit small tokenizer models. For large/experimental tokenizers, ignore them.

Option A: ignore generated tokenizers by default:

```gitignore
tokenizers/
```

Option B: keep the directory but ignore generated files:

```gitignore
tokenizers/**/*.model
tokenizers/**/*.vocab
tokenizers/**/*.train.txt
```

Option C: commit curated tokenizer models and ignore temporary files only:

```gitignore
tokenizers/**/*.train.txt
```

Recommended for Vibroso right now:

```gitignore
tokenizers/**/*.train.txt
```

and commit the `.model` / `.vocab` only if intentionally curated.

---

## Acceptance Criteria

This change is complete when:

1. `sentencepiece` is listed in `requirements.txt`.
2. `TrainConfig` accepts `tokenizer_kind: sentencepiece`.
3. Config supports:
   - `tokenizer_model_path`
   - `tokenizer_prefix`
   - `tokenizer_model_type`
4. `SentencePieceTokenizer` implements:
   - `encode`
   - `decode`
   - `vocab_size`
   - `to_dict`
   - `from_dict`
   - optionally `encode_to_pieces`
5. `build_tokenizer(...)` can train or load a SentencePiece tokenizer.
6. `tokenizer_from_dict(...)` can restore a SentencePiece tokenizer from checkpoint metadata.
7. Training works with:

   ```yaml
   tokenizer_kind: sentencepiece
   tokenizer_vocab_size: 4096
   tokenizer_model_type: bpe
   tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
   tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model
   ```

8. Generation works from a checkpoint trained with SentencePiece.
9. Benchmark tokenizer script shows meaningful pieces and improved compression.
10. Existing `char` and custom `subword` tokenizers still work.

---

## Validation Commands

### Install dependency

```powershell
pip install sentencepiece
```

### Train with SentencePiece

```powershell
python scripts/train.py --config configs/sentencepiece.yaml
```

### Inspect tokenizer

```powershell
python scripts/benchmark_tokenizer.py --config configs/sentencepiece.yaml
```

Expected sample tokenization should show subword pieces, not character-level pieces.

### Generate

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/latest.pt `
  --prompt "Tortilla`nA tortilla is" `
  --temperature 0.5 `
  --top-k 30 `
  --max-new-tokens 120
```

---

## Recommended Initial Config

Add:

```text
configs/sentencepiece.yaml
```

with:

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
```

---

## Notes

SentencePiece will make tokenizer training and dataset encoding much more practical for large corpora, but it changes token IDs and vocabulary. Therefore:

> Any checkpoint trained with the custom tokenizer is incompatible with SentencePiece tokenization.

Retrain from scratch after switching tokenizers.
