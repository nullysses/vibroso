from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.config import Config
from toy_llm.dataset import TextDataset, load_corpus
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.sampling import count_parameters
from toy_llm.tokenizer import build_tokenizer, tokenizer_from_dict
from toy_llm.train_loop import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT.")
    parser.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    torch.manual_seed(config.seed)
    device = get_device(config.device)

    optimizer_state = None
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        config = Config.from_dict(checkpoint["config"])
        device = get_device(config.device)
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        start_step = int(checkpoint.get("step", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")

    text = load_corpus(config)
    if not args.resume:
        tokenizer = build_tokenizer(
            text,
            kind=config.tokenizer_kind,
            vocab_size=config.tokenizer_vocab_size,
        )
    dataset = TextDataset.from_text(
        text,
        tokenizer,
        config.train_split,
        config.block_size,
        config.batch_size,
        device,
    )
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

    print(f"device: {device}")
    print(f"dataset chars: {len(text)}")
    print(f"dataset tokens: {len(tokenizer.encode(text))}")
    print(f"tokenizer: {config.tokenizer_kind}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"parameters: {count_parameters(model):,}")

    train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        config=config,
        start_step=start_step,
        optimizer_state_dict=optimizer_state,
    )


if __name__ == "__main__":
    main()
