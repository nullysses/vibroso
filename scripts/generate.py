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
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a tiny GPT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = Config.from_dict(checkpoint["config"])
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        n_layer=config.model.n_layer,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
