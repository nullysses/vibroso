from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from toy_llm.checkpoint import save_checkpoint
from toy_llm.config import TrainConfig
from toy_llm.dataset import TextDataset
from toy_llm.model import TinyGPT
from toy_llm.sampling import make_optimizer
from toy_llm.tokenizer import Tokenizer


def use_mixed_precision(device: torch.device) -> bool:
    return device.type == "cuda"


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    dataset: TextDataset,
    eval_iters: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = dataset.get_batch(split)
            with torch.autocast(
                device_type=dataset.device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision(dataset.device),
            ):
                _, loss = model(xb, yb)
            assert loss is not None
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


def train(
    model: TinyGPT,
    dataset: TextDataset,
    tokenizer: Tokenizer,
    config: TrainConfig,
    start_step: int = 0,
    optimizer_state_dict: dict | None = None,
    metrics_path: str | Path | None = None,
    metrics_metadata: dict[str, object] | None = None,
) -> None:
    optimizer = make_optimizer(model, config.learning_rate, config.weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    use_amp = use_mixed_precision(dataset.device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    latest_losses: dict[str, float | None] = {"train": None, "val": None}
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(metrics_path) if metrics_path is not None else None
    if metrics_file is not None:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text("", encoding="utf-8")
    metrics_base = dict(metrics_metadata or {})
    last_metrics_time = time.perf_counter()
    last_metrics_step = start_step

    for step in range(start_step, config.max_steps + 1):
        if step % config.eval_interval == 0:
            now = time.perf_counter()
            trained_steps = step - last_metrics_step
            elapsed = now - last_metrics_time
            tokens_per_sec = (
                trained_steps * config.batch_size * config.block_size / elapsed
                if trained_steps > 0 and elapsed > 0
                else 0.0
            )
            losses = estimate_loss(model, dataset, config.eval_iters)
            latest_losses = losses
            train_ppl = torch.exp(torch.tensor(losses["train"])).item()
            val_ppl = torch.exp(torch.tensor(losses["val"])).item()
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}"
            )
            if metrics_file is not None:
                record = {
                    **metrics_base,
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_perplexity": val_ppl,
                    "tokens_per_sec": tokens_per_sec,
                }
                with metrics_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
            last_metrics_time = time.perf_counter()
            last_metrics_step = step

        if step % config.checkpoint_interval == 0 and step != start_step:
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                config,
                tokenizer,
                step,
                latest_losses["train"],
                latest_losses["val"],
            )

        if step == config.max_steps:
            break

        xb, yb = dataset.get_batch("train")
        with torch.autocast(
            device_type=dataset.device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            _, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    save_checkpoint(
        checkpoint_dir / "latest.pt",
        model,
        optimizer,
        config,
        tokenizer,
        config.max_steps,
        latest_losses["train"],
        latest_losses["val"],
    )
