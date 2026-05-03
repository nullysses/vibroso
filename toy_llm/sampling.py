from __future__ import annotations

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
