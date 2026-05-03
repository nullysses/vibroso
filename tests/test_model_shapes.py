import pytest
import torch

from toy_llm.model import CausalSelfAttentionHead, TinyGPT


def make_model() -> TinyGPT:
    return TinyGPT(
        vocab_size=11,
        block_size=8,
        n_embd=16,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )


def test_forward_returns_expected_logits_and_scalar_loss():
    model = make_model()
    idx = torch.randint(0, 11, (3, 8))
    targets = torch.randint(0, 11, (3, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 8, 11)
    assert loss is not None
    assert loss.ndim == 0


def test_forward_without_targets_returns_no_loss():
    model = make_model()
    logits, loss = model(torch.randint(0, 11, (2, 5)))
    assert logits.shape == (2, 5, 11)
    assert loss is None


def test_forward_rejects_sequence_longer_than_block_size():
    model = make_model()
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 11, (1, 9)))


def test_attention_mask_shape_and_future_positions():
    head = CausalSelfAttentionHead(n_embd=16, head_size=4, block_size=8, dropout=0.0)
    assert head.tril.shape == (8, 8)
    assert head.tril[0, 1] == 0
    assert head.tril[7, 0] == 1
    out = head(torch.randn(2, 6, 16))
    assert out.shape == (2, 6, 4)
