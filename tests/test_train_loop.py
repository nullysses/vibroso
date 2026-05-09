import torch

from toy_llm.config import ModelConfig, TrainConfig
from toy_llm.dataset import TextDataset
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import CharTokenizer
from toy_llm.train_loop import train, use_mixed_precision


def test_mixed_precision_is_cuda_only():
    assert not use_mixed_precision(torch.device("cpu"))
    assert not use_mixed_precision(torch.device("mps"))
    assert use_mixed_precision(torch.device("cuda"))


def test_train_loop_runs_on_cpu_with_amp_disabled(tmp_path):
    torch.manual_seed(0)
    text = "abcdefghijklmnopqrstuvwxyz" * 8
    tokenizer = CharTokenizer.from_text(text)
    device = torch.device("cpu")
    dataset = TextDataset.from_text(
        text,
        tokenizer,
        train_split=0.9,
        block_size=8,
        batch_size=2,
        device=device,
    )
    config = TrainConfig(
        checkpoint_dir=str(tmp_path),
        max_steps=1,
        eval_interval=1,
        eval_iters=1,
        checkpoint_interval=1,
        batch_size=2,
        block_size=8,
        model=ModelConfig(n_embd=8, n_head=2, n_layer=1, dropout=0.0),
    )
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        n_layer=config.model.n_layer,
        dropout=config.model.dropout,
    )

    train(model, dataset, tokenizer, config)

    assert (tmp_path / "latest.pt").exists()
