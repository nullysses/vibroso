# Toy LLM — Functional and Technical Specification

## 1. Executive Summary

This project builds a small, functional language model from scratch using Python and PyTorch. The first version is a character-level autoregressive Transformer trained on a local text corpus. It supports training, checkpointing, text generation, and basic evaluation.

The goal is not to compete with production LLMs. The goal is to understand the core moving parts of an LLM by implementing them directly: tokenization, embeddings, positional information, causal self-attention, Transformer blocks, training loops, loss computation, checkpointing, and sampling.

---

# Functional Specification

## 2. Product Goal

Build a minimal language model that can:

1. Load a text dataset.
2. Tokenize text into discrete IDs.
3. Train a small autoregressive Transformer.
4. Save and load model checkpoints.
5. Generate text from a prompt.
6. Report basic training and validation metrics.

The model should be small enough to train on a local machine, preferably with CPU support and optional GPU acceleration.

---

## 3. Target User

Primary user:

- A software engineer who wants to understand LLM internals by building one.

Secondary users:

- Interview prep candidates exploring ML systems.
- Backend engineers learning PyTorch.
- Developers interested in model-serving architecture.

---

## 4. Scope

### 4.1 In Scope for Version 1

Version 1 implements:

- Character-level tokenizer.
- Local plain-text dataset ingestion.
- Train/validation split.
- Tiny GPT-style Transformer decoder.
- Causal self-attention.
- Multi-head attention.
- Feed-forward layers.
- Layer normalization.
- Residual connections.
- Cross-entropy loss.
- AdamW optimizer.
- Periodic evaluation.
- Checkpoint save/load.
- CLI-based text generation.
- Configurable hyperparameters.

### 4.2 Out of Scope for Version 1

Version 1 does not implement:

- BPE, WordPiece, or SentencePiece tokenization.
- Distributed training.
- RLHF, DPO, or instruction tuning.
- Retrieval-augmented generation.
- Fine-tuning on chat data.
- Model quantization.
- FlashAttention or custom CUDA kernels.
- Production web serving.
- Multi-GPU support.
- Safety filtering.
- Dataset scraping.

---

## 5. User Stories

### 5.1 Dataset Loading

As a user, I want to point the program at a text file so that the model can train on it.

Acceptance criteria:

- The program accepts a dataset path.
- The dataset is loaded as raw text.
- The program reports dataset size in characters and tokens.
- Empty or missing files produce clear errors.

### 5.2 Tokenization

As a user, I want the project to tokenize text into IDs so that the model can consume it.

Acceptance criteria:

- The tokenizer builds a vocabulary from the dataset.
- Each unique character receives an integer ID.
- The tokenizer supports `encode(text) -> list[int]`.
- The tokenizer supports `decode(ids) -> str`.
- Encoding and decoding a known string round-trips correctly.

### 5.3 Training

As a user, I want to train the model so that it learns to predict the next character.

Acceptance criteria:

- The training command starts from a config file or default config.
- The model trains for a configurable number of steps.
- Training loss is printed periodically.
- Validation loss is computed periodically.
- Training can run on CPU or GPU.
- The program saves checkpoints.

### 5.4 Generation

As a user, I want to provide a prompt and generate continuation text.

Acceptance criteria:

- The generation command accepts a prompt string.
- The generation command loads a saved checkpoint.
- The user can configure max generated tokens.
- The user can configure temperature.
- The user can configure top-k sampling.
- The program prints generated text to stdout.

### 5.5 Checkpointing

As a user, I want to save and restore model state so that I do not need to retrain from scratch.

Acceptance criteria:

- Checkpoints include model weights.
- Checkpoints include optimizer state.
- Checkpoints include config.
- Checkpoints include tokenizer vocabulary.
- Training can resume from a checkpoint.
- Generation can load from a checkpoint.

### 5.6 Evaluation

As a user, I want basic metrics so that I can see whether the model is learning.

Acceptance criteria:

- Training loss is reported.
- Validation loss is reported.
- Optional perplexity is derived from loss.
- Evaluation runs without gradient updates.

---

## 6. Functional Requirements

## FR-1: Dataset Ingestion

The system shall read a local UTF-8 plain-text file.

Inputs:

- `dataset_path`: path to `.txt` file.

Outputs:

- Raw text string.
- Dataset statistics.

Validation:

- File must exist.
- File must be non-empty.
- File must be decodable as UTF-8.

---

## FR-2: Character Vocabulary Construction

The system shall build a sorted character vocabulary from the dataset.

Inputs:

- Raw dataset text.

Outputs:

- `stoi`: string-to-integer mapping.
- `itos`: integer-to-string mapping.
- `vocab_size`.

---

## FR-3: Encode and Decode

The tokenizer shall encode strings into integer token IDs and decode token IDs back into strings.

Behavior:

- Unknown characters at generation time should either raise a clear error or map to a configurable unknown token.
- For Version 1, the preferred behavior is to raise a clear error because the character vocabulary is dataset-derived.

---

## FR-4: Train/Validation Split

The system shall split tokenized data into train and validation partitions.

Default split:

- 90% train.
- 10% validation.

Configurable:

- `train_split` between `0.5` and `0.99`.

---

## FR-5: Batch Sampling

The system shall sample random batches of contiguous token sequences.

For each batch:

- Input `x`: tokens from position `i` to `i + block_size - 1`.
- Target `y`: tokens from position `i + 1` to `i + block_size`.

Shape:

- `x`: `[batch_size, block_size]`.
- `y`: `[batch_size, block_size]`.

---

## FR-6: Model Forward Pass

The model shall accept a tensor of token IDs and return logits over the vocabulary.

Input:

- `idx`: tensor shaped `[batch_size, sequence_length]`.

Output:

- `logits`: tensor shaped `[batch_size, sequence_length, vocab_size]`.

Optional output:

- `loss`, if targets are provided.

---

## FR-7: Autoregressive Causal Constraint

The model shall not attend to future tokens during training or generation.

Requirement:

- Attention mask must prevent token position `t` from attending to positions greater than `t`.

---

## FR-8: Training Loop

The system shall train the model using next-token prediction.

Core loop:

1. Sample batch.
2. Run forward pass.
3. Compute cross-entropy loss.
4. Clear gradients.
5. Backpropagate.
6. Update weights.
7. Periodically evaluate.
8. Periodically checkpoint.

---

## FR-9: Text Generation

The system shall generate tokens autoregressively.

Generation loop:

1. Encode prompt.
2. Feed current context into model.
3. Extract logits for final position.
4. Apply temperature.
5. Optionally apply top-k filtering.
6. Sample next token.
7. Append token to context.
8. Repeat until max tokens reached.
9. Decode full sequence.

---

## FR-10: CLI Commands

The system shall provide CLI commands for common operations.

Required commands:

```bash
python train.py --config configs/tiny.yaml
python generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --max-new-tokens 300
```

Optional command:

```bash
python inspect_tokenizer.py --dataset data/input.txt
```

---

## 7. Non-Functional Requirements

## NFR-1: Simplicity

The codebase should prioritize readability over maximum performance.

## NFR-2: Reproducibility

The system should support setting a random seed.

## NFR-3: Local Execution

The system should run locally on:

- CPU-only machines.
- CUDA-enabled GPUs, if available.
- Apple Silicon MPS, if available.

## NFR-4: Modularity

Tokenizer, model, dataset, training, generation, and configuration should live in separate modules.

## NFR-5: Inspectability

The code should make it easy to inspect:

- Batch shapes.
- Loss curves.
- Generated samples.
- Model parameter count.

## NFR-6: Fail-Fast Behavior

Invalid config values, missing files, or incompatible checkpoint/config combinations should produce clear errors.

---

## 8. Success Criteria

The project is successful when:

1. A user can train the model on a small text corpus.
2. Training loss decreases over time.
3. Validation loss is reported.
4. A checkpoint is saved.
5. The saved checkpoint can generate text.
6. The generated text resembles the training corpus at least superficially.
7. The code is small enough that one person can understand the whole system.

---

## 9. Example User Flow

1. User places a text file at:

```bash
data/input.txt
```

2. User runs:

```bash
python train.py --config configs/tiny.yaml
```

3. System prints:

```text
vocab_size: 65
parameters: 10.7M
step 0: train loss 4.23, val loss 4.22
step 500: train loss 2.71, val loss 2.89
step 1000: train loss 2.31, val loss 2.55
```

4. User generates text:

```bash
python generate.py --checkpoint checkpoints/latest.pt --prompt "KING:" --temperature 0.8 --max-new-tokens 500
```

5. System prints generated text.

---

# Technical Specification

## 10. Architecture Overview

The project is a local Python application with four core layers:

```text
CLI layer
  -> configuration layer
  -> data/tokenization layer
  -> model/training/generation layer
  -> checkpointing layer
```

The model architecture is a decoder-only Transformer similar in structure to a very small GPT model.

---

## 11. Proposed Repository Structure

```text
toy_llm/
  README.md
  requirements.txt
  pyproject.toml

  configs/
    tiny.yaml
    small.yaml

  data/
    input.txt

  checkpoints/
    .gitkeep

  toy_llm/
    __init__.py
    config.py
    tokenizer.py
    dataset.py
    model.py
    train_loop.py
    checkpoint.py
    sampling.py
    device.py

  scripts/
    train.py
    generate.py
    inspect_tokenizer.py

  tests/
    test_tokenizer.py
    test_dataset.py
    test_model_shapes.py
    test_sampling.py
```

---

## 12. Runtime Dependencies

Required:

```text
python >= 3.10
torch
pyyaml
```

Optional:

```text
tqdm
numpy
matplotlib
```

Version 1 can avoid heavy dependencies beyond PyTorch and PyYAML.

---

## 13. Configuration

Configuration should be represented as a dataclass and loaded from YAML.

Example config:

```yaml
dataset_path: data/input.txt
checkpoint_dir: checkpoints

seed: 1337
device: auto

train_split: 0.9
batch_size: 32
block_size: 128

max_steps: 5000
eval_interval: 500
eval_iters: 100
checkpoint_interval: 1000
learning_rate: 0.0003
weight_decay: 0.1

model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.2
```

Validation rules:

- `batch_size > 0`.
- `block_size > 0`.
- `n_embd % n_head == 0`.
- `0 < train_split < 1`.
- `learning_rate > 0`.
- `max_steps > 0`.

---

## 14. Tokenizer Design

### 14.1 Class: `CharTokenizer`

Responsibilities:

- Build vocabulary.
- Encode strings.
- Decode IDs.
- Serialize vocabulary.
- Restore vocabulary from checkpoint.

Suggested interface:

```python
class CharTokenizer:
    def __init__(self, chars: list[str]):
        ...

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        ...
```

### 14.2 Vocabulary Ordering

Use sorted unique characters for deterministic vocabulary construction:

```python
chars = sorted(list(set(text)))
```

---

## 15. Dataset and Batch Sampling

### 15.1 Encoded Dataset

After tokenization, represent the full corpus as a 1D PyTorch tensor:

```python
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
```

Split:

```python
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]
```

### 15.2 Batch Function

Suggested interface:

```python
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
```

---

## 16. Model Architecture

## 16.1 High-Level Model

The model is a decoder-only Transformer.

Pipeline:

```text
token IDs
  -> token embeddings
  -> positional embeddings
  -> Transformer blocks
  -> final layer norm
  -> linear output head
  -> logits over vocabulary
```

---

## 16.2 Main Model Class

Suggested class name:

```python
class TinyGPT(nn.Module):
    ...
```

Constructor inputs:

- `vocab_size`
- `block_size`
- `n_embd`
- `n_head`
- `n_layer`
- `dropout`

Forward signature:

```python
def forward(
    self,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ...
```

Inputs:

- `idx`: `[B, T]`
- `targets`: `[B, T]`, optional

Outputs:

- `logits`: `[B, T, vocab_size]`
- `loss`: scalar tensor or `None`

---

## 16.3 Token Embeddings

Use:

```python
nn.Embedding(vocab_size, n_embd)
```

Input shape:

```text
[B, T]
```

Output shape:

```text
[B, T, C]
```

Where:

- `B = batch_size`
- `T = sequence length`
- `C = n_embd`

---

## 16.4 Positional Embeddings

For Version 1, use learned positional embeddings:

```python
nn.Embedding(block_size, n_embd)
```

Position IDs:

```python
pos = torch.arange(0, T, device=device)
```

Combined embedding:

```python
x = token_embedding(idx) + position_embedding(pos)
```

---

## 16.5 Causal Self-Attention Head

Each attention head computes:

```text
Q = XWq
K = XWk
V = XWv
attention_scores = QK^T / sqrt(head_size)
masked_scores = causal_mask(attention_scores)
attention_weights = softmax(masked_scores)
out = attention_weights V
```

Mask:

- Lower-triangular matrix of shape `[T, T]`.
- Future positions are filled with negative infinity before softmax.

Suggested implementation:

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

---

## 16.6 Multi-Head Attention

Multi-head attention runs several attention heads in parallel and concatenates their outputs.

```python
head_outputs = [head(x) for head in self.heads]
out = torch.cat(head_outputs, dim=-1)
out = self.proj(out)
```

Constraint:

```python
head_size = n_embd // n_head
```

---

## 16.7 Feed-Forward Network

Each Transformer block includes a position-wise feed-forward network.

Suggested implementation:

```python
nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

GELU may be used instead of ReLU for a more GPT-like architecture.

Recommended:

```python
nn.GELU()
```

---

## 16.8 Transformer Block

Use pre-norm Transformer blocks for training stability.

Shape-preserving transformation:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Components:

- LayerNorm 1.
- Multi-head causal self-attention.
- Residual connection.
- LayerNorm 2.
- Feed-forward network.
- Residual connection.

---

## 16.9 Output Head

Final projection:

```python
lm_head = nn.Linear(n_embd, vocab_size)
```

Output:

```text
[B, T, vocab_size]
```

---

## 17. Loss Function

Use cross-entropy loss over next-token prediction.

Because PyTorch expects logits shaped `[N, C]` and targets shaped `[N]`, reshape:

```python
B, T, C = logits.shape
logits = logits.view(B * T, C)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

---

## 18. Optimizer

Use AdamW.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)
```

Optional later:

- Learning-rate warmup.
- Cosine decay.
- Gradient clipping.

For Version 1, fixed learning rate is acceptable.

---

## 19. Training Loop Details

Pseudo-code:

```python
for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % checkpoint_interval == 0:
        save_checkpoint(...)
```

Evaluation mode:

```python
model.eval()
with torch.no_grad():
    ...
model.train()
```

---

## 20. Generation Algorithm

Method on model:

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

Edge cases:

- If `temperature <= 0`, reject config or use greedy decoding explicitly.
- If `top_k > vocab_size`, clamp or reject.
- Prompt length greater than block size is allowed; only the latest `block_size` tokens are used during each generation step.

---

## 21. Checkpoint Format

Checkpoint should be a dictionary:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config_dict,
    "tokenizer": tokenizer.to_dict(),
    "step": step,
    "train_loss": train_loss,
    "val_loss": val_loss,
}
```

Save:

```python
torch.save(checkpoint, path)
```

Load:

```python
checkpoint = torch.load(path, map_location=device)
```

---

## 22. Device Selection

Device selection should support:

1. CUDA if available.
2. MPS if available.
3. CPU fallback.

Suggested behavior:

```python
def get_device(config_device: str) -> torch.device:
    if config_device != "auto":
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

---

## 23. CLI Design

## 23.1 Training CLI

Command:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Arguments:

- `--config`: path to YAML config.
- `--resume`: optional checkpoint path.

Behavior:

- Load config.
- Load dataset.
- Build or restore tokenizer.
- Build model.
- Print parameter count.
- Train.
- Save checkpoints.

---

## 23.2 Generation CLI

Command:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

Arguments:

- `--checkpoint`
- `--prompt`
- `--max-new-tokens`
- `--temperature`
- `--top-k`
- `--device`

Behavior:

- Load checkpoint.
- Restore config.
- Restore tokenizer.
- Restore model weights.
- Encode prompt.
- Generate token IDs.
- Decode and print text.

---

## 24. Testing Plan

## 24.1 Tokenizer Tests

Test cases:

- Vocabulary construction is deterministic.
- `decode(encode(text)) == text` for known text.
- Unknown character behavior is explicit.
- Serialized tokenizer can be restored.

## 24.2 Dataset Tests

Test cases:

- Train/validation split sizes are correct.
- Batch shapes are `[B, T]`.
- Targets are shifted by one token.

## 24.3 Model Shape Tests

Test cases:

- Forward pass returns logits of shape `[B, T, vocab_size]`.
- Loss is scalar when targets are provided.
- Loss is `None` when targets are omitted.
- Model rejects sequence length greater than block size in forward pass.

## 24.4 Attention Mask Tests

Test cases:

- Causal mask shape is valid.
- Future tokens are masked.
- Attention output shape matches input embedding shape.

## 24.5 Generation Tests

Test cases:

- Generation returns original prompt plus new tokens.
- Generation respects `max_new_tokens`.
- Top-k sampling handles small vocabularies.

---

## 25. Implementation Milestones

## Milestone 1: Minimal Bigram Model

Goal:

- Prove dataset, tokenizer, training, and generation loop.

Deliverables:

- Character tokenizer.
- Batch sampler.
- Bigram language model.
- Training script.
- Generation script.

Why:

- A bigram model is easier to debug than a Transformer.
- It validates the pipeline before adding attention.

---

## Milestone 2: Single-Head Self-Attention

Goal:

- Add causal attention and verify shape behavior.

Deliverables:

- Attention head module.
- Causal masking.
- Tests for attention shapes and masking.

---

## Milestone 3: Multi-Head Transformer

Goal:

- Implement full tiny GPT-style model.

Deliverables:

- Multi-head attention.
- Feed-forward network.
- Transformer block.
- Stacked blocks.
- Final layer norm and LM head.

---

## Milestone 4: Checkpointing and Resumption

Goal:

- Make training resumable and generation checkpoint-based.

Deliverables:

- Save checkpoint.
- Load checkpoint.
- Resume training.
- Generate from saved checkpoint.

---

## Milestone 5: Polish and Observability

Goal:

- Make the project pleasant to use.

Deliverables:

- Better CLI output.
- Parameter count.
- Loss reporting.
- README.
- Example config.
- Basic tests.

---

## 26. Default Hyperparameter Profiles

## 26.1 Tiny CPU-Friendly Config

```yaml
batch_size: 16
block_size: 64
max_steps: 2000
learning_rate: 0.001
model:
  n_embd: 128
  n_head: 4
  n_layer: 2
  dropout: 0.1
```

Expected behavior:

- Trains on CPU.
- Generates crude but recognizable corpus-like text.
- Good for debugging.

---

## 26.2 Small GPU-Friendly Config

```yaml
batch_size: 64
block_size: 256
max_steps: 10000
learning_rate: 0.0003
model:
  n_embd: 384
  n_head: 6
  n_layer: 6
  dropout: 0.2
```

Expected behavior:

- Better text quality.
- Requires more memory.
- Suitable for CUDA or MPS.

---

## 27. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Loss does not decrease | High | Start with bigram model and tiny dataset |
| Model overfits quickly | Medium | Use validation loss and dropout |
| Training too slow on CPU | Medium | Provide tiny config |
| Shape bugs in attention | High | Add shape tests early |
| Bad generated text discourages user | Medium | Set expectations: toy model, small corpus |
| Tokenizer cannot encode prompt chars | Medium | Warn or reject unknown characters clearly |

---

## 28. Key Design Decisions

## 28.1 Character-Level Tokenization First

Decision:

- Use character-level tokenization for Version 1.

Reason:

- It is transparent, deterministic, and easy to implement.
- It avoids the complexity of BPE.
- It keeps the focus on the Transformer.

Tradeoff:

- Generated text quality is worse than subword tokenization.
- Longer sequences are required to capture semantic structure.

---

## 28.2 Decoder-Only Transformer

Decision:

- Use a GPT-style decoder-only architecture.

Reason:

- It directly matches autoregressive next-token prediction.
- It is conceptually aligned with most modern LLMs.
- Encoder-decoder architecture is unnecessary for this toy goal.

---

## 28.3 Learned Positional Embeddings

Decision:

- Use learned positional embeddings.

Reason:

- They are simple to implement.
- They work well enough for fixed context windows.

Tradeoff:

- They do not extrapolate naturally beyond the trained block size.

---

## 28.4 Pre-Norm Blocks

Decision:

- Use pre-layer normalization inside Transformer blocks.

Reason:

- More stable for training deeper Transformers.
- Common in modern GPT-like architectures.

---

## 29. Future Extensions

After Version 1, possible extensions include:

1. BPE tokenizer.
2. Dataset streaming.
3. Larger corpus support.
4. Learning-rate scheduling.
5. Gradient clipping.
6. Weight tying between embedding and output head.
7. FlashAttention-compatible implementation.
8. LoRA fine-tuning.
9. Instruction-tuned toy assistant.
10. Simple FastAPI inference server.
11. Web UI for generation.
12. Quantized inference.
13. ONNX export.
14. Rust or Go inference wrapper.

---

## 30. Recommended Build Order

The recommended implementation order is:

1. `tokenizer.py`
2. `dataset.py`
3. `model.py` with bigram baseline
4. `train.py`
5. `generate.py`
6. Replace bigram model with attention head
7. Add multi-head attention
8. Add Transformer block
9. Add stacked Transformer
10. Add checkpointing
11. Add tests
12. Add README

This order minimizes debugging ambiguity. The data pipeline should work before the Transformer is introduced.

---

## 31. Minimum Viable Version

The smallest useful version contains:

```text
CharTokenizer
get_batch
TinyGPT
train.py
generate.py
checkpoint save/load
```

Anything beyond that is useful but not required for first success.

---

## 32. Definition of Done

Version 1 is done when the following command sequence works:

```bash
python scripts/train.py --config configs/tiny.yaml
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "hello" --max-new-tokens 300
```

And:

- Training loss decreases.
- Validation loss is reported.
- A checkpoint is saved.
- Generated output is non-random-looking relative to the corpus.
- The codebase remains understandable to a single engineer.

