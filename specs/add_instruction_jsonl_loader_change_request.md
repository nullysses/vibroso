# Change Request: Add JSONL Instruction Dataset Loader

## Summary

Add support for instruction fine-tuning datasets stored as **JSONL** files.

Vibroso currently trains as a base language model by reading raw text and learning next-token prediction over that text. This change adds a dataset loading path for instruction-response examples, so a file like:

```jsonl
{"instruction":"What is a tortilla?","response":"A tortilla is a thin flatbread made from corn or wheat flour."}
{"instruction":"What is artificial intelligence?","response":"Artificial intelligence is the field of building systems that perform tasks associated with intelligence."}
```

can be converted into formatted training text like:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
```

This is the first infrastructure step toward an instruction fine-tuning stage.

---

## Goals

### Primary goals

- Add `dataset_kind: instruction_jsonl`.
- Load one JSON object per line from a `.jsonl` file.
- Validate required fields.
- Convert instruction-response examples into a consistent chat-style text format.
- Reuse the existing tokenizer and training loop.
- Keep existing raw text, URL list, and Wikipedia dataset behavior unchanged.
- Support small overfit/debug instruction datasets first.

### Non-goals

- Do not implement full RLHF, DPO, or preference training.
- Do not implement multi-turn conversation trees yet.
- Do not implement masking loss only on assistant tokens yet.
- Do not download Dolly/OASST automatically in this change.
- Do not require a new model architecture.
- Do not make Vibroso a production chatbot.

---

## Dataset Format

### Required JSONL format

Each line must be a JSON object with at least:

```json
{
  "instruction": "What is a tortilla?",
  "response": "A tortilla is a thin flatbread made from corn or wheat flour."
}
```

### Optional field: `context`

Some instruction datasets include extra context/input.

Supported shape:

```json
{
  "instruction": "Summarize this text.",
  "context": "Tortillas are thin flatbreads made from corn or wheat...",
  "response": "A tortilla is a thin flatbread, often made from corn or wheat."
}
```

### Field rules

Required:

```text
instruction: non-empty string
response: non-empty string
```

Optional:

```text
context: string, may be empty or absent
```

Invalid rows should raise a clear error by default.

---

## Formatting

Add a formatter function:

```python
def format_instruction_example(
    instruction: str,
    response: str,
    context: str | None = None,
) -> str:
    ...
```

Recommended output without context:

```text
<|user|>
{instruction}
<|assistant|>
{response}
<|end|>
```

Recommended output with context:

```text
<|user|>
{instruction}

Context:
{context}
<|assistant|>
{response}
<|end|>
```

Examples should be separated by a blank line in the final corpus text.

Final concatenated corpus:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>

<|user|>
What is artificial intelligence?
<|assistant|>
Artificial intelligence is the field of building systems that perform tasks associated with intelligence.
<|end|>
```

---

## Special Tokens

Instruction fine-tuning should use consistent role markers:

```text
<|user|>
<|assistant|>
<|end|>
```

For SentencePiece runs, these should ideally be added as user-defined symbols in a later change.

For this change, it is acceptable if SentencePiece tokenizes them as normal text, but the loader should use the exact same marker strings consistently.

Recommended future SentencePiece config field:

```yaml
tokenizer_user_defined_symbols:
  - "<|user|>"
  - "<|assistant|>"
  - "<|end|>"
```

This field does not need to be implemented in this change unless convenient.

---

## Config Changes

Update `TrainConfig.validate()` to allow:

```yaml
dataset_kind: instruction_jsonl
```

Current accepted values likely include:

```text
text
url_list
wikipedia_titles
```

New accepted values:

```text
text
url_list
wikipedia_titles
instruction_jsonl
```

Example config:

```yaml
dataset_path: data/instruction_examples.jsonl
dataset_kind: instruction_jsonl

tokenizer_kind: sentencepiece
tokenizer_vocab_size: 4096
tokenizer_model_type: bpe
tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model
tokenizer_train_chars: 1000000

checkpoint_dir: checkpoints/instruct

seed: 1337
device: cuda

train_split: 0.9
batch_size: 16
block_size: 256

max_steps: 1000
eval_interval: 100
eval_iters: 20
checkpoint_interval: 250
learning_rate: 0.00005
weight_decay: 0.1

model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.1
  mlp_type: swiglu
```

---

## Dataset Loader Changes

Update `toy_llm/dataset.py`.

Add:

```python
def load_instruction_jsonl(path: str | Path) -> str:
    ...
```

### Suggested implementation

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_non_empty_string(row: dict[str, Any], key: str, line_no: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Invalid instruction JSONL row at line {line_no}: "
            f"field {key!r} must be a non-empty string"
        )
    return value.strip()


def format_instruction_example(
    instruction: str,
    response: str,
    context: str | None = None,
) -> str:
    instruction = instruction.strip()
    response = response.strip()
    context = context.strip() if context else ""

    if context:
        return (
            "<|user|>\n"
            f"{instruction}\n\n"
            "Context:\n"
            f"{context}\n"
            "<|assistant|>\n"
            f"{response}\n"
            "<|end|>\n"
        )

    return (
        "<|user|>\n"
        f"{instruction}\n"
        "<|assistant|>\n"
        f"{response}\n"
        "<|end|>\n"
    )


def load_instruction_jsonl(path: str | Path) -> str:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Instruction JSONL file not found: {dataset_path}")

    examples: list[str] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {dataset_path}:{line_no}: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid instruction JSONL row at line {line_no}: "
                    "expected a JSON object"
                )

            instruction = _require_non_empty_string(row, "instruction", line_no)
            response = _require_non_empty_string(row, "response", line_no)

            context_value = row.get("context")
            if context_value is not None and not isinstance(context_value, str):
                raise ValueError(
                    f"Invalid instruction JSONL row at line {line_no}: "
                    "field 'context' must be a string when present"
                )

            examples.append(
                format_instruction_example(
                    instruction=instruction,
                    response=response,
                    context=context_value,
                )
            )

    if not examples:
        raise ValueError(f"Instruction JSONL file contains no examples: {dataset_path}")

    return "\n".join(examples)
```

---

## `load_corpus` Integration

Update `load_corpus(config)` so that:

```python
if config.dataset_kind == "instruction_jsonl":
    return load_instruction_jsonl(config.dataset_path)
```

This should happen before URL/Wikipedia-specific handling.

Expected behavior:

| `dataset_kind` | Loader |
|---|---|
| `text` | existing raw UTF-8 text loader |
| `url_list` | existing URL fetch/cache loader |
| `wikipedia_titles` | existing Wikipedia fetch/cache loader |
| `instruction_jsonl` | new JSONL instruction loader |

---

## Training From Existing Base Checkpoint

Instruction fine-tuning usually starts from a base LM checkpoint.

The current `scripts/train.py --resume checkpoints/latest.pt` behavior may resume both model and optimizer state. For instruction fine-tuning, that is not ideal: you usually want to load model weights/tokenizer/config compatibility, but use a new optimizer and a lower learning rate.

This change may keep the simple path first:

```powershell
python scripts/train.py --config configs/instruction_tiny.yaml --resume checkpoints/base_latest.pt
```

But recommended future behavior is a separate flag:

```powershell
python scripts/train.py `
  --config configs/instruction_tiny.yaml `
  --init-from checkpoints/base_latest.pt
```

Semantics:

```text
--resume
  Continue an interrupted run, including optimizer state and step.

--init-from
  Initialize model weights/tokenizer from a base checkpoint, but start a new training run with a fresh optimizer.
```

Implementing `--init-from` is useful but can be a follow-up change if needed.

---

## Generation Prompt Format

After training on instruction JSONL, generation should use:

```text
<|user|>
What is a tortilla?
<|assistant|>
```

Example command:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.4 `
  --top-k 20 `
  --max-new-tokens 120
```

Expected output shape:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour, commonly used in Mexican cuisine.
<|end|>
```

---

## Stop Token Support

This loader introduces the `<|end|>` marker, but generation may continue after it.

A follow-up change should add stop-string support:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --stop "<|end|>"
```

This is not required for the JSONL loader itself, but it is recommended soon after.

---

## Tests

Add tests for the loader.

Suggested test file:

```text
tests/test_instruction_jsonl.py
```

### Test: basic loading

```python
def test_load_instruction_jsonl_basic(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text(
        '{"instruction":"What is a tortilla?","response":"A flatbread."}\n',
        encoding="utf-8",
    )

    text = load_instruction_jsonl(path)

    assert "<|user|>" in text
    assert "What is a tortilla?" in text
    assert "<|assistant|>" in text
    assert "A flatbread." in text
    assert "<|end|>" in text
```

### Test: context field

```python
def test_load_instruction_jsonl_with_context(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text(
        (
            '{"instruction":"Summarize.","context":"A tortilla is a flatbread.",'
            '"response":"A tortilla is a flatbread."}\n'
        ),
        encoding="utf-8",
    )

    text = load_instruction_jsonl(path)

    assert "Context:" in text
    assert "A tortilla is a flatbread." in text
```

### Test: invalid JSON

```python
def test_load_instruction_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_instruction_jsonl(path)
```

### Test: missing required field

```python
def test_load_instruction_jsonl_rejects_missing_response(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction":"What is a tortilla?"}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_instruction_jsonl(path)
```

### Test: empty dataset

```python
def test_load_instruction_jsonl_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_instruction_jsonl(path)
```

---

## Optional Utility Script

Add:

```text
scripts/preview_instruction_dataset.py
```

Purpose:

```text
Load an instruction JSONL file and print the formatted first N examples.
```

Example:

```powershell
python scripts/preview_instruction_dataset.py `
  --dataset data/instruction_examples.jsonl `
  --limit 3
```

Suggested output:

```text
=== Example 1 ===
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
```

This is useful to verify formatting before training.

---

## Suggested Initial Files

Add:

```text
data/instruction_examples.example.jsonl
```

with a few small examples:

```jsonl
{"instruction":"What is a tortilla?","response":"A tortilla is a thin flatbread made from corn or wheat flour, commonly used in Mexican cuisine."}
{"instruction":"What is artificial intelligence?","response":"Artificial intelligence is the field of building systems that perform tasks associated with human intelligence."}
{"instruction":"What is Vibroso?","response":"Vibroso is an experimental toy language model project built for learning how LLM components work."}
```

Do not commit large instruction datasets unless intentionally curated.

---

## `.gitignore` Guidance

If real instruction datasets are generated or downloaded, decide whether to commit them.

Recommended:

```gitignore
data/instruction_*.jsonl
!data/instruction_examples.example.jsonl
```

This keeps the small example file but avoids accidentally committing large datasets.

If the user wants to commit the 123-example seed file, use a clear curated name:

```text
data/instruction_seed_123.jsonl
```

and document that it is a small synthetic overfit/debug corpus.

---

## Acceptance Criteria

This change is complete when:

1. `TrainConfig.validate()` accepts `dataset_kind: instruction_jsonl`.
2. `load_instruction_jsonl(...)` exists.
3. `load_instruction_jsonl(...)` validates required fields.
4. `load_instruction_jsonl(...)` supports optional `context`.
5. `load_corpus(config)` routes `instruction_jsonl` to the new loader.
6. The formatted output uses the exact markers:
   - `<|user|>`
   - `<|assistant|>`
   - `<|end|>`
7. Existing dataset kinds still work unchanged.
8. Unit tests cover valid rows, context rows, invalid JSON, missing fields, and empty files.
9. A small example JSONL file is added or documented.
10. Training can run with:

   ```yaml
   dataset_kind: instruction_jsonl
   dataset_path: data/instruction_examples.example.jsonl
   ```

---

## Validation Commands

Run tests:

```powershell
pytest
```

Preview the loaded corpus manually, if preview script exists:

```powershell
python scripts/preview_instruction_dataset.py `
  --dataset data/instruction_examples.example.jsonl `
  --limit 3
```

Train on the tiny instruction dataset:

```powershell
python scripts/train.py --config configs/instruction_tiny.yaml
```

Generate with instruction format:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.4 `
  --top-k 20 `
  --max-new-tokens 120
```

---

## Notes

This is primarily an infrastructure change. The first goal is to make Vibroso capable of overfitting a small instruction dataset and producing answer-shaped text.

General QA quality should not be expected until:

- the base LM is stronger,
- instruction examples are larger and cleaner,
- special role markers are handled well by the tokenizer,
- generation supports stopping at `<|end|>`,
- and benchmark prompts show improved instruction-following behavior.
