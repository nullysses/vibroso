# Change Request: Add Stop-String Support to Generation

## Summary

Add stop-string support to text generation so Vibroso can stop generation when it emits a marker such as:

```text
<|end|>
```

This is especially useful for instruction-tuned checkpoints trained on examples formatted as:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
```

Without stop-string support, generation may continue beyond `<|end|>` and produce extra text, another user turn, or unrelated continuation.

---

## Goals

### Primary goals

- Add `--stop` to `scripts/generate.py`.
- Support one or more stop strings.
- Stop generation after the first emitted stop string.
- Remove the stop string from displayed output by default, or support a flag to keep it.
- Preserve existing generation behavior when no stop string is provided.
- Work with any tokenizer backend, including SentencePiece.

### Non-goals

- Do not change model architecture.
- Do not change training.
- Do not require instruction tuning.
- Do not implement beam search.
- Do not implement advanced chat templates.
- Do not implement token-level EOS training in this change.

---

## CLI Changes

Update `scripts/generate.py` to accept:

```powershell
--stop "<|end|>"
```

Optionally allow repeated stops:

```powershell
--stop "<|end|>" --stop "<|user|>"
```

Suggested argparse:

```python
parser.add_argument(
    "--stop",
    action="append",
    default=None,
    help="Stop generation when this string appears. Can be provided multiple times.",
)
parser.add_argument(
    "--keep-stop",
    action="store_true",
    help="Keep the stop string in the printed output instead of trimming it.",
)
```

Example:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct_seed123/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.35 `
  --top-k 20 `
  --max-new-tokens 120 `
  --stop "<|end|>"
```

---

## Config Changes

Update `InferenceConfig` to include:

```python
stop: list[str] | None = None
keep_stop: bool = False
```

If YAML config is used:

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
```

If `InferenceConfig` should remain simple, it is acceptable to implement stop handling only in `scripts/generate.py` first. However, config support is preferred.

---

## Behavior

Generation should stop when the decoded output contains any stop string.

Given output:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
Extra unwanted text...
```

With:

```text
--stop "<|end|>"
```

Printed output should be:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
```

If `--keep-stop` is set, printed output should be:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
```

---

## Implementation Approach

### Simple string-level approach

The simplest implementation is to generate token-by-token as usual, decode after each new token or small step, and check whether any stop string appears.

Current model-level `generate(...)` returns a full tensor after `max_new_tokens`.

For stop support, either:

1. Add stop handling inside `TinyGPT.generate(...)`, or
2. Add a script-level generation loop in `scripts/generate.py`.

Recommended first version: add support inside `TinyGPT.generate(...)` only if you can keep it tokenizer-agnostic cleanly. Since the model does not know how to decode tokens, script-level handling may be cleaner.

### Recommended script-level helper

Add helper:

```python
def truncate_at_stop(text: str, stop_strings: list[str], keep_stop: bool = False) -> tuple[str, bool]:
    best_index: int | None = None
    best_stop: str | None = None

    for stop in stop_strings:
        idx = text.find(stop)
        if idx == -1:
            continue
        if best_index is None or idx < best_index:
            best_index = idx
            best_stop = stop

    if best_index is None or best_stop is None:
        return text, False

    end = best_index + len(best_stop) if keep_stop else best_index
    return text[:end], True
```

### More efficient generation loop

Instead of calling `model.generate(...)` once for all tokens, `scripts/generate.py` can run generation one token at a time and check decoded output:

```python
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

for _ in range(inference_config.max_new_tokens):
    idx = model.generate(
        idx,
        max_new_tokens=1,
        temperature=inference_config.temperature,
        top_k=inference_config.top_k,
    )

    decoded = tokenizer.decode(idx[0].tolist())
    truncated, found_stop = truncate_at_stop(
        decoded,
        stop_strings=inference_config.stop or [],
        keep_stop=inference_config.keep_stop,
    )
    if found_stop:
        print(truncated)
        return

print(tokenizer.decode(idx[0].tolist()))
```

This is simple but may be slower because it repeatedly calls `model.generate(...)`. Since `model.generate(...)` already handles KV cache internally per call, repeatedly calling it with `max_new_tokens=1` may not preserve the cache across calls. For small instruction outputs, this is acceptable as a first version.

### Better model API approach

Add a lower-level generation method that yields tokens or accepts a callback. Example future direction:

```python
def generate_stream(...):
    yield idx_next
```

Not required for this change.

---

## Tokenizer Considerations

Stop strings are checked after decoding text, not by token IDs.

This is tokenizer-agnostic and works with:

```text
char
custom subword
sentencepiece
```

Caveat: some tokenizers may normalize whitespace. Therefore, stop markers should be simple and stable:

```text
<|end|>
```

Avoid stop strings that depend on exact newline preservation.

---

## Interaction With Prompt

By default, stop detection should check the full decoded text, including prompt. This can be a problem if the prompt already contains the stop string.

For instruction prompts, the prompt usually does not contain `<|end|>`, so this is fine.

More robust behavior:

1. Decode the full text.
2. Remove or ignore the original prompt prefix for stop detection.
3. Apply stop detection only to generated suffix.

Recommended implementation:

```python
prompt_text = inference_config.prompt
decoded = tokenizer.decode(idx[0].tolist())

generated_suffix = decoded[len(prompt_text):]
truncated_suffix, found_stop = truncate_at_stop(
    generated_suffix,
    stop_strings,
    keep_stop=keep_stop,
)
output_text = prompt_text + truncated_suffix
```

But because tokenizers may normalize prompt whitespace, `decoded.startswith(prompt_text)` may not always hold exactly. A simpler robust first version can search the full decoded text, with a note that stop strings should not appear in the prompt.

---

## Tests

Add tests for stop-string truncation logic.

Suggested test file:

```text
tests/test_generation_stop.py
```

### Test: truncate without stop

```python
def test_truncate_at_stop_without_match():
    text, stopped = truncate_at_stop("hello world", ["<|end|>"])
    assert text == "hello world"
    assert stopped is False
```

### Test: truncate with stop removed

```python
def test_truncate_at_stop_removes_stop_by_default():
    text, stopped = truncate_at_stop("answer<|end|>extra", ["<|end|>"])
    assert text == "answer"
    assert stopped is True
```

### Test: truncate with stop kept

```python
def test_truncate_at_stop_can_keep_stop():
    text, stopped = truncate_at_stop("answer<|end|>extra", ["<|end|>"], keep_stop=True)
    assert text == "answer<|end|>"
    assert stopped is True
```

### Test: earliest stop wins

```python
def test_truncate_at_stop_uses_earliest_stop():
    text, stopped = truncate_at_stop("a<stop2>b<stop1>c", ["<stop1>", "<stop2>"])
    assert text == "a"
    assert stopped is True
```

---

## Acceptance Criteria

This change is complete when:

1. `scripts/generate.py` accepts `--stop`.
2. `--stop` can be provided at least once.
3. Existing generation works unchanged when no stop string is provided.
4. Generation output is truncated at the first stop string.
5. `--keep-stop` optionally preserves the stop string in output.
6. Stop logic works with SentencePiece checkpoints.
7. Tests cover stop-string truncation behavior.
8. README or generation help documents use with instruction-tuned checkpoints.

---

## Validation Commands

Generate with stop marker:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct_seed123/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.35 `
  --top-k 20 `
  --max-new-tokens 120 `
  --stop "<|end|>"
```

Generate while preserving stop marker:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct_seed123/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.35 `
  --top-k 20 `
  --max-new-tokens 120 `
  --stop "<|end|>" `
  --keep-stop
```

Expected shape without `--keep-stop`:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
```

Expected shape with `--keep-stop`:

```text
<|user|>
What is a tortilla?
<|assistant|>
A tortilla is a thin flatbread made from corn or wheat flour.
<|end|>
```

---

## Notes

Stop-string support does not make the model better at instruction following by itself. It only makes instruction-tuned output cleaner by cutting off generation at a known boundary.

This should be implemented after or alongside instruction fine-tuning support.
