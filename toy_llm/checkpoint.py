from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from toy_llm.config import TrainConfig
from toy_llm.tokenizer import Tokenizer


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    tokenizer: Tokenizer,
    step: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)
