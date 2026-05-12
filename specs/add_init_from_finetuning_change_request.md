# Change Request: Add `--init-from` for Fine-Tuning From a Base Checkpoint

## Summary

Add a new training mode, `--init-from`, that initializes a new training run from an existing checkpoint's model weights and tokenizer, while using the newly supplied config, dataset, optimizer, learning rate, checkpoint directory, and step counter.

This is needed for the workflow:

```text
Wikipedia/base LM training
→ save base checkpoint
→ instruction/QA fine-tuning from base checkpoint
→ save instruction-tuned checkpoint
```

Current `--resume` behavior is correct for continuing an interrupted run, but it is not correct for instruction fine-tuning because it reloads the checkpoint's original config, optimizer state, and step.

---

## Goals

### Primary goals

- Add `--init-from checkpoints/base/latest.pt` to `scripts/train.py`.
- Load model weights from the base checkpoint.
- Load tokenizer from the base checkpoint.
- Use the new config passed through `--config`.
- Use the new dataset from the new config.
- Start training at step `0`.
- Use a fresh optimizer.
- Save checkpoints to the new config's `checkpoint_dir`.
- Preserve existing `--resume` behavior unchanged.

### Non-goals

- Do not remove `--resume`.
- Do not reuse optimizer state during `--init-from`.
- Do not silently allow architecture-incompatible checkpoints.
- Do not retrain the tokenizer during `--init-from`.
- Do not implement instruction JSONL loading in this change; that is already supported separately.
- Do not implement generation stop tokens in this change.

---

## Current Behavior

Current `scripts/train.py` supports:

```powershell
python scripts/train.py --config configs/wiki.yaml --resume checkpoints/latest.pt
```

When `--resume` is used, it:

1. Loads the checkpoint.
2. Replaces the supplied config with the checkpoint config.
3. Loads the checkpoint tokenizer.
4. Loads model weights.
5. Loads optimizer state.
6. Starts from the checkpoint step.

This is correct for interrupted training.

It is not correct for fine-tuning on a new dataset.

---

## Desired Behavior

Add:

```powershell
python scripts/train.py `
  --config configs/instruction_seed.yaml `
  --init-from checkpoints/base/latest.pt
```

This should:

1. Load `configs/instruction_seed.yaml`.
2. Load the base checkpoint.
3. Load tokenizer from the base checkpoint.
4. Instantiate the model using the new config.
5. Validate that the new config is compatible with the base checkpoint architecture.
6. Load `model_state_dict` from the base checkpoint.
7. Do **not** load optimizer state.
8. Start at step `0`.
9. Load the instruction dataset from the new config.
10. Save new checkpoints to the new config's `checkpoint_dir`.

---

## CLI Changes

Update `scripts/train.py`:

```python
parser.add_argument(
    "--init-from",
    default=None,
    help="Initialize model weights and tokenizer from a checkpoint, but start a new training run.",
)
```

Reject using both:

```powershell
--resume
--init-from
```

at the same time.

Suggested validation:

```python
if args.resume and args.init_from:
    raise ValueError("Use either --resume or --init-from, not both.")
```

---

## Semantics: `--resume` vs `--init-from`

| Behavior | `--resume` | `--init-from` |
|---|---:|---:|
| Load model weights | Yes | Yes |
| Load tokenizer | Yes | Yes |
| Load checkpoint config | Yes | Only for compatibility checks |
| Use supplied config | No/current behavior | Yes |
| Load optimizer state | Yes | No |
| Continue old step | Yes | No |
| Start from step 0 | No | Yes |
| Use new dataset | No | Yes |
| Use new learning rate | No | Yes |
| Use new checkpoint dir | No | Yes |

---

## Compatibility Checks

During `--init-from`, compare the new config to the checkpoint config.

The following should match exactly:

```text
model.n_embd
model.n_head
model.n_layer
model.mlp_type
block_size
```

The tokenizer comes from the checkpoint, so `tokenizer_kind`, `tokenizer_vocab_size`, and SentencePiece paths in the new config should not be used to rebuild a tokenizer.

The effective `vocab_size` comes from the checkpoint tokenizer.

Recommended compatibility helper:

```python
def validate_init_from_compatibility(new_config: TrainConfig, base_config: TrainConfig) -> None:
    if new_config.block_size != base_config.block_size:
        raise ValueError(
            "init-from requires matching block_size: "
            f"new={new_config.block_size}, checkpoint={base_config.block_size}"
        )

    if new_config.model != base_config.model:
        raise ValueError(
            "init-from requires matching model config. "
            f"new={new_config.model}, checkpoint={base_config.model}"
        )
```

A slightly more flexible later version may allow `batch_size`, `learning_rate`, `max_steps`, `checkpoint_dir`, and dataset fields to differ. For this first version, only architecture-critical fields should be enforced.

Allowed to differ:

```text
dataset_path
dataset_kind
checkpoint_dir
seed
device
train_split
batch_size
max_steps
eval_interval
eval_iters
checkpoint_interval
learning_rate
weight_decay
corpus_cache_path
```

---

## Implementation Sketch

Current structure roughly does:

```python
config = TrainConfig.from_yaml(args.config)
optimizer_state = None
start_step = 0

if args.resume:
    checkpoint = load_checkpoint(args.resume, device)
    config = TrainConfig.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    start_step = int(checkpoint.get("step", 0))
    optimizer_state = checkpoint.get("optimizer_state_dict")
```

Proposed structure:

```python
config = TrainConfig.from_yaml(args.config)
torch.manual_seed(config.seed)
device = get_device(config.device)

checkpoint = None
optimizer_state = None
start_step = 0
tokenizer = None

if args.resume and args.init_from:
    raise ValueError("Use either --resume or --init-from, not both.")

if args.resume:
    checkpoint = load_checkpoint(args.resume, device)
    config = TrainConfig.from_dict(checkpoint["config"])
    device = get_device(config.device)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    start_step = int(checkpoint.get("step", 0))
    optimizer_state = checkpoint.get("optimizer_state_dict")

elif args.init_from:
    checkpoint = load_checkpoint(args.init_from, device)
    base_config = TrainConfig.from_dict(checkpoint["config"])
    validate_init_from_compatibility(config, base_config)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    start_step = 0
    optimizer_state = None
```

Then later:

```python
if tokenizer is None:
    tokenizer = build_tokenizer(...)
```

And after model construction:

```python
if checkpoint is not None:
    model.load_state_dict(checkpoint["model_state_dict"])
```

---

## Tokenizer Behavior

For `--init-from`, do **not** call `build_tokenizer(...)`.

Always load tokenizer from the base checkpoint:

```python
tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
```

Reason:

```text
The model weights are tied to the original token IDs.
```

If the instruction dataset is encoded with a different tokenizer, the loaded model weights become semantically invalid.

---

## Optimizer Behavior

For `--init-from`, use a fresh optimizer.

Do not load:

```python
checkpoint["optimizer_state_dict"]
```

Reason:

Instruction fine-tuning uses a different dataset and usually a lower learning rate. Reusing AdamW moments from base training can make fine-tuning less predictable.

Expected call into `train(...)`:

```python
train(
    model=model,
    dataset=dataset,
    tokenizer=tokenizer,
    config=config,
    start_step=0,
    optimizer_state_dict=None,
    ...
)
```

---

## Example Config: Instruction Fine-Tune

```yaml
dataset_path: data/instruction_seed_123.jsonl
dataset_kind: instruction_jsonl

# These tokenizer fields may remain for documentation, but init-from uses
# the tokenizer stored in the base checkpoint.
tokenizer_kind: sentencepiece
tokenizer_vocab_size: 4096
tokenizer_model_path: tokenizers/sentencepiece/wiki_4096.model
tokenizer_prefix: tokenizers/sentencepiece/wiki_4096
tokenizer_model_type: bpe
tokenizer_train_chars: 1000000

checkpoint_dir: checkpoints/instruct_seed123

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

Run:

```powershell
python scripts/train.py `
  --config configs/instruction_seed.yaml `
  --init-from checkpoints/base/latest.pt
```

---

## Tests

Add or update tests for training initialization behavior.

### Test: cannot use `--resume` and `--init-from` together

Expected:

```text
ValueError or parser error
```

### Test: init-from does not load optimizer state

Mock or inspect that `optimizer_state_dict=None` is passed to `train(...)`.

### Test: init-from starts at step 0

Expected:

```python
start_step == 0
```

### Test: init-from uses supplied config

Use a base checkpoint config with:

```yaml
dataset_kind: wikipedia_titles
checkpoint_dir: checkpoints/base
learning_rate: 0.0003
```

Use supplied instruction config with:

```yaml
dataset_kind: instruction_jsonl
checkpoint_dir: checkpoints/instruct
learning_rate: 0.00005
```

Expected: instruction config values are used.

### Test: init-from rejects incompatible model config

For example:

```text
base model.n_embd = 256
new model.n_embd = 384
```

Expected: clear `ValueError`.

### Test: init-from rejects incompatible block_size

For example:

```text
base block_size = 256
new block_size = 128
```

Expected: clear `ValueError`.

---

## Acceptance Criteria

This change is complete when:

1. `scripts/train.py` accepts `--init-from`.
2. `--resume` behavior remains unchanged.
3. `--resume` and `--init-from` cannot be used together.
4. `--init-from` loads model weights from checkpoint.
5. `--init-from` loads tokenizer from checkpoint.
6. `--init-from` uses the supplied `--config` for dataset, checkpoint dir, learning rate, max steps, etc.
7. `--init-from` does not load optimizer state.
8. `--init-from` starts from step `0`.
9. `--init-from` validates architecture compatibility.
10. Fine-tuning from a base checkpoint to an `instruction_jsonl` dataset works.
11. The resulting checkpoint is saved to the new config's `checkpoint_dir`.
12. Existing base training and resume training still work.

---

## Validation Commands

Base training:

```powershell
python scripts/train.py `
  --config configs/sentencepiece_swiglu.yaml
```

Fine-tuning from base checkpoint:

```powershell
python scripts/train.py `
  --config configs/instruction_seed.yaml `
  --init-from checkpoints/base/latest.pt
```

Generate from instruction checkpoint:

```powershell
python scripts/generate.py `
  --checkpoint checkpoints/instruct_seed123/latest.pt `
  --prompt "<|user|>`nWhat is a tortilla?`n<|assistant|>`n" `
  --temperature 0.35 `
  --top-k 20 `
  --max-new-tokens 120
```

---

## Notes

This change is the key bridge between base LM training and instruction fine-tuning.

After this is implemented, Vibroso can support the intended workflow:

```text
train base LM
keep base checkpoint
init-from base checkpoint
fine-tune on QA/instruction JSONL
validate instruction-style generation
```
