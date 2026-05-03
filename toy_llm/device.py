from __future__ import annotations

import torch


def get_device(config_device: str = "auto") -> torch.device:
    if config_device != "auto":
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
