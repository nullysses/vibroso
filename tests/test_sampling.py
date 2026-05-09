import torch

from toy_llm.model import TinyGPT
from toy_llm.sampling import count_parameters


def test_generation_returns_prompt_plus_new_tokens():
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=5,
        block_size=4,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    idx = torch.tensor([[1, 2]])
    out = model.generate(idx, max_new_tokens=3, top_k=2)
    assert out.shape == (1, 5)
    assert torch.equal(out[:, :2], idx)


def test_generation_clamps_top_k_to_vocab_size():
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=5,
        block_size=4,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    out = model.generate(torch.tensor([[1]]), max_new_tokens=2, top_k=100)
    assert out.shape == (1, 3)


def test_cached_generation_matches_uncached_generation():
    model = TinyGPT(
        vocab_size=7,
        block_size=5,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()
    idx = torch.tensor([[1, 2, 3]])

    torch.manual_seed(123)
    cached = model.generate(idx, max_new_tokens=4, top_k=3, use_kv_cache=True)
    torch.manual_seed(123)
    uncached = model.generate(idx, max_new_tokens=4, top_k=3, use_kv_cache=False)

    assert torch.equal(cached, uncached)


def test_count_parameters_is_positive():
    model = TinyGPT(
        vocab_size=5,
        block_size=4,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    assert count_parameters(model) > 0
