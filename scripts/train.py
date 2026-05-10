from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.config import TrainConfig
from toy_llm.dataset import TextDataset, load_corpus
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.sampling import count_parameters
from toy_llm.tokenizer import build_tokenizer, tokenizer_from_dict
from toy_llm.train_loop import train


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _time_start() -> float:
    return time.perf_counter()


def _print_elapsed(label: str, start: float) -> None:
    print(f"{label} done in {_format_elapsed(time.perf_counter() - start)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT.")
    parser.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--metrics", default=None, help="Optional JSONL metrics output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)
    torch.manual_seed(config.seed)
    device = get_device(config.device)

    optimizer_state = None
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        config = TrainConfig.from_dict(checkpoint["config"])
        device = get_device(config.device)
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        start_step = int(checkpoint.get("step", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")

    phase_start = _time_start()
    text = load_corpus(config)
    _print_elapsed("loading corpus", phase_start)

    if not args.resume:
        print("building tokenizer...")
        phase_start = _time_start()
        tokenizer = build_tokenizer(
            text,
            kind=config.tokenizer_kind,
            vocab_size=config.tokenizer_vocab_size,
            max_train_chars=config.tokenizer_train_chars,
        )
        _print_elapsed("building tokenizer", phase_start)
    print("encoding dataset...")
    phase_start = _time_start()
    dataset = TextDataset.from_text(
        text,
        tokenizer,
        config.train_split,
        config.block_size,
        config.batch_size,
        device,
    )
    _print_elapsed("encoding dataset", phase_start)

    phase_start = _time_start()
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        n_layer=config.model.n_layer,
        dropout=config.model.dropout,
    ).to(device)

    if args.resume:
        model.load_state_dict(checkpoint["model_state_dict"])
    _print_elapsed("initializing model", phase_start)

    print(f"device: {device}")
    phase_start = _time_start()
    dataset_tokens = len(dataset.train_data) + len(dataset.val_data)
    print(f"dataset chars: {len(text)}")
    print(f"dataset tokens: {dataset_tokens}")
    print(f"tokenizer: {config.tokenizer_kind}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "merge_count"):
        print(f"subword merges: {tokenizer.merge_count}")
    parameters = count_parameters(model)
    print(f"parameters: {parameters:,}")
    _print_elapsed("startup stats", phase_start)

    metrics_metadata = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_kind": config.tokenizer_kind,
        "tokenizer_train_chars": config.tokenizer_train_chars,
        "dataset_chars": len(text),
        "dataset_tokens": dataset_tokens,
        "chars_per_token": len(text) / max(dataset_tokens, 1),
        "parameters": parameters,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "model": {
            "n_embd": config.model.n_embd,
            "n_head": config.model.n_head,
            "n_layer": config.model.n_layer,
            "dropout": config.model.dropout,
        },
    }

    train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        config=config,
        start_step=start_step,
        optimizer_state_dict=optimizer_state,
        metrics_path=args.metrics,
        metrics_metadata=metrics_metadata,
    )


if __name__ == "__main__":
    main()
