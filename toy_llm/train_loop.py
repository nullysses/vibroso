from __future__ import annotations

from pathlib import Path

import torch

from toy_llm.checkpoint import save_checkpoint
from toy_llm.config import TrainConfig
from toy_llm.dataset import TextDataset
from toy_llm.model import TinyGPT
from toy_llm.sampling import make_optimizer
from toy_llm.tokenizer import Tokenizer


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
) -> None:
    optimizer = make_optimizer(model, config.learning_rate, config.weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    model.train()
    latest_losses: dict[str, float | None] = {"train": None, "val": None}
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(start_step, config.max_steps + 1):
        if step % config.eval_interval == 0:
            losses = estimate_loss(model, dataset, config.eval_iters)
            latest_losses = losses
            train_ppl = torch.exp(torch.tensor(losses["train"])).item()
            val_ppl = torch.exp(torch.tensor(losses["val"])).item()
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}"
            )

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
        _, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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
