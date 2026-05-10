# Change Request: Apply Learned Subword Merges During Encoding

## Summary

The current `SubwordTokenizer` appears to build a larger vocabulary with merged tokens, but `encode()` is still emitting one token per character. As a result, the model is effectively training and generating as a character-level model even when `tokenizer_kind: subword` is configured.

Update `SubwordTokenizer.encode()` so it applies the learned pair merges before converting pieces to token IDs.

---

## Evidence

Running:

```powershell
python scripts/inspect_checkpoint_tokenizer.py `
  --checkpoint checkpoints/latest.pt `
  --sample "A tortilla is a traditional Mexican flatbread."
```

currently produces output like:

```text
tokenizer class: SubwordTokenizer
vocab size: 1069
token count: 46
pieces:
     35: 'A'
      2: ' '
     86: 't'
     81: 'o'
     84: 'r'
     86: 't'
     75: 'i'
     78: 'l'
     78: 'l'
     67: 'a'
      ...
```

The tokenizer has a vocabulary larger than the base character set, but the encoded output is still character-level.

---

## Current Behavior

For this sample:

```text
A tortilla is a traditional Mexican flatbread.
```

the tokenizer emits one token per character:

```text
'A', ' ', 't', 'o', 'r', 't', 'i', 'l', 'l', 'a', ...
```

This means the learned merged tokens are not being used by `encode()`.

---

## Expected Behavior

`SubwordTokenizer.encode()` should use the learned merge rules so common words and subword fragments are represented by multi-character tokens.

The exact pieces depend on the trained vocabulary and merge order, but acceptable output should look more like one of these:

```text
'A', ' tortilla', ' is', ' a', ' traditional', ' Mexican', ' flatbread', '.'
```

or:

```text
'A', ' tort', 'illa', ' is', ' a', ' tradition', 'al', ' Mexican', ' flat', 'bread', '.'
```

The sample sentence should encode to significantly fewer than 46 tokens.

---

## Required Change

Modify `SubwordTokenizer.encode()` so that it:

1. Starts with the input text as a list of character pieces.
2. Applies learned pair merges in the same order they were learned during tokenizer training.
3. Converts the resulting merged pieces into token IDs using `stoi`.

Conceptual implementation:

```python
def encode(self, text: str) -> list[int]:
    pieces = list(text)

    for left, right, merged in self.merges:
        new_pieces = []
        i = 0

        while i < len(pieces):
            if (
                i + 1 < len(pieces)
                and pieces[i] == left
                and pieces[i + 1] == right
            ):
                new_pieces.append(merged)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1

        pieces = new_pieces

    return [self.stoi[piece] for piece in pieces]
```

Adjust field names to match the current implementation.

---

## Implementation Notes

### Preserve Round-Trip Behavior

The tokenizer must continue to satisfy:

```python
decoded = tokenizer.decode(tokenizer.encode(text))
assert decoded == text
```

for text covered by the tokenizer vocabulary.

### Preserve Merge Order

Merge order matters. The encoder should apply merges in the same sequence produced during training. Applying merges out of order can produce different tokenization from the learned vocabulary.

### Do Not Reuse Old Checkpoints

After changing `encode()`, old checkpoints should be treated as incompatible. The same text will now produce different token IDs, so previous model weights no longer correspond to the new tokenization.

Retrain from scratch after this change.

---

## Acceptance Criteria

After the change:

1. `SubwordTokenizer.encode()` applies learned merges.
2. `decode(encode(text)) == text` still holds for known text.
3. The diagnostic script shows multi-character pieces for common text.
4. The sentence:

   ```text
   A tortilla is a traditional Mexican flatbread.
   ```

   encodes to significantly fewer than 46 tokens.
5. Training with `tokenizer_kind: subword` produces a dataset token count lower than the character-level tokenizer.
6. Generated samples no longer fail primarily at the spelling/word-formation level when using a properly trained subword tokenizer.
7. Old checkpoints are deleted or ignored and the model is retrained from scratch.

---

## Validation Command

After retraining from scratch, run:

```powershell
python scripts/inspect_checkpoint_tokenizer.py `
  --checkpoint checkpoints/latest.pt `
  --sample "A tortilla is a traditional Mexican flatbread."
```

Expected result:

```text
tokenizer class: SubwordTokenizer
token count: 10-20  # approximate target
pieces:
  multi-character tokens should appear here
```

The exact token count may vary, but the output should not be purely character-level.

---

## Suggested Follow-Up

After this fix is validated, increase the tokenizer vocabulary size for the Wikipedia corpus:

```yaml
tokenizer_kind: subword
tokenizer_vocab_size: 2048
```

Then consider testing:

```yaml
tokenizer_vocab_size: 4096
```

if the corpus is large enough and the model capacity can support the larger output vocabulary.
