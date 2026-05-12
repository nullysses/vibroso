import argparse

import pytest
import torch

from scripts.train import load_training_state, validate_init_from_compatibility
from toy_llm.checkpoint import save_checkpoint
from toy_llm.config import ModelConfig, TrainConfig
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import CharTokenizer


def _model_config(**overrides):
    values = {
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 1,
        "dropout": 0.0,
        "mlp_type": "gelu",
    }
    values.update(overrides)
    return ModelConfig(**values)


def _train_config(**overrides):
    values = {
        "dataset_path": "data/base.txt",
        "dataset_kind": "text",
        "checkpoint_dir": "checkpoints/base",
        "device": "cpu",
        "batch_size": 2,
        "block_size": 8,
        "max_steps": 4,
        "eval_interval": 1,
        "eval_iters": 1,
        "checkpoint_interval": 1,
        "learning_rate": 3e-4,
        "model": _model_config(),
    }
    values.update(overrides)
    return TrainConfig(**values)


def _write_checkpoint(path, config):
    tokenizer = CharTokenizer.from_text("abc ")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        n_layer=config.model.n_layer,
        dropout=config.model.dropout,
        mlp_type=config.model.mlp_type,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    save_checkpoint(path, model, optimizer, config, tokenizer, step=7)
    return tokenizer


def _args(**overrides):
    values = {"resume": None, "init_from": None, "metrics": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_init_from_rejects_resume_at_same_time():
    config = _train_config()

    with pytest.raises(ValueError, match="Use either --resume or --init-from"):
        load_training_state(
            _args(resume="checkpoints/base.pt", init_from="checkpoints/base.pt"),
            config,
            torch.device("cpu"),
        )


def test_init_from_uses_supplied_config_without_optimizer_or_step(tmp_path):
    base_config = _train_config(
        dataset_kind="wikipedia_titles",
        checkpoint_dir="checkpoints/base",
        learning_rate=3e-4,
    )
    checkpoint_path = tmp_path / "base.pt"
    base_tokenizer = _write_checkpoint(checkpoint_path, base_config)
    finetune_config = _train_config(
        dataset_path="data/instruction.jsonl",
        dataset_kind="instruction_jsonl",
        checkpoint_dir="checkpoints/instruct",
        learning_rate=5e-5,
        max_steps=100,
    )

    loaded_config, _, checkpoint, tokenizer, start_step, optimizer_state = load_training_state(
        _args(init_from=str(checkpoint_path)),
        finetune_config,
        torch.device("cpu"),
    )

    assert loaded_config == finetune_config
    assert loaded_config.dataset_kind == "instruction_jsonl"
    assert loaded_config.checkpoint_dir == "checkpoints/instruct"
    assert loaded_config.learning_rate == 5e-5
    assert start_step == 0
    assert optimizer_state is None
    assert checkpoint is not None
    assert tokenizer is not None
    assert tokenizer.to_dict() == base_tokenizer.to_dict()


def test_resume_keeps_checkpoint_config_optimizer_and_step(tmp_path):
    base_config = _train_config(
        dataset_kind="wikipedia_titles",
        checkpoint_dir="checkpoints/base",
        learning_rate=3e-4,
    )
    checkpoint_path = tmp_path / "base.pt"
    _write_checkpoint(checkpoint_path, base_config)
    supplied_config = _train_config(
        dataset_path="data/instruction.jsonl",
        dataset_kind="instruction_jsonl",
        checkpoint_dir="checkpoints/instruct",
        learning_rate=5e-5,
    )

    loaded_config, _, checkpoint, tokenizer, start_step, optimizer_state = load_training_state(
        _args(resume=str(checkpoint_path)),
        supplied_config,
        torch.device("cpu"),
    )

    assert loaded_config == base_config
    assert start_step == 7
    assert optimizer_state is not None
    assert checkpoint is not None
    assert tokenizer is not None


def test_init_from_rejects_incompatible_model_config():
    base_config = _train_config()
    finetune_config = _train_config(model=_model_config(n_embd=12))

    with pytest.raises(ValueError, match="init-from requires matching model config"):
        validate_init_from_compatibility(finetune_config, base_config)


def test_init_from_rejects_incompatible_block_size():
    base_config = _train_config(block_size=8)
    finetune_config = _train_config(block_size=16)

    with pytest.raises(ValueError, match="init-from requires matching block_size"):
        validate_init_from_compatibility(finetune_config, base_config)
