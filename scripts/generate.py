from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.config import InferenceConfig, TrainConfig
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import tokenizer_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a tiny GPT checkpoint.")
    parser.add_argument("--config", default=None, help="Optional YAML inference config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument("--prompt", default=None, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_config = (
        InferenceConfig.from_yaml(args.config) if args.config else InferenceConfig()
    ).with_overrides(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    if inference_config.seed is not None:
        torch.manual_seed(inference_config.seed)
    device = get_device(inference_config.device)
    checkpoint = load_checkpoint(inference_config.checkpoint, device)
    train_config = TrainConfig.from_dict(checkpoint["config"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=train_config.block_size,
        n_embd=train_config.model.n_embd,
        n_head=train_config.model.n_head,
        n_layer=train_config.model.n_layer,
        dropout=train_config.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(inference_config.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=inference_config.max_new_tokens,
        temperature=inference_config.temperature,
        top_k=inference_config.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
