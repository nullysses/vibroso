from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        if self.n_embd <= 0:
            raise ValueError("model.n_embd must be > 0")
        if self.n_head <= 0:
            raise ValueError("model.n_head must be > 0")
        if self.n_layer <= 0:
            raise ValueError("model.n_layer must be > 0")
        if self.n_embd % self.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be >= 0 and < 1")


@dataclass
class Config:
    dataset_path: str = "data/input.txt"
    dataset_kind: str = "text"
    tokenizer_kind: str = "subword"
    tokenizer_vocab_size: int = 256
    corpus_cache_path: str | None = None
    fetch_timeout: float = 20.0
    user_agent: str = "toy-llm/0.1 academic corpus builder"
    wikipedia_language: str = "en"
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337
    device: str = "auto"
    train_split: float = 0.9
    batch_size: int = 16
    block_size: int = 64
    max_steps: int = 2000
    eval_interval: int = 100
    eval_iters: int = 20
    checkpoint_interval: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must contain a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        values = dict(data)
        model_data = values.pop("model", {})
        if isinstance(model_data, ModelConfig):
            model = model_data
        elif isinstance(model_data, dict):
            model = ModelConfig(**model_data)
        else:
            raise ValueError("model config must be a mapping")
        config = cls(**values, model=model)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.dataset_kind not in {"text", "url_list", "wikipedia_titles"}:
            raise ValueError("dataset_kind must be 'text', 'url_list', or 'wikipedia_titles'")
        if self.tokenizer_kind not in {"subword", "char"}:
            raise ValueError("tokenizer_kind must be 'subword' or 'char'")
        if self.tokenizer_vocab_size <= 0:
            raise ValueError("tokenizer_vocab_size must be > 0")
        if self.fetch_timeout <= 0:
            raise ValueError("fetch_timeout must be > 0")
        if not self.wikipedia_language:
            raise ValueError("wikipedia_language must be non-empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if not 0.5 <= self.train_split <= 0.99:
            raise ValueError("train_split must be between 0.5 and 0.99")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0")
        if self.eval_iters <= 0:
            raise ValueError("eval_iters must be > 0")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0")
        self.model.validate()
