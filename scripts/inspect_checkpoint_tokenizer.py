from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.device import get_device
from toy_llm.tokenizer import tokenizer_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tokenizer pieces stored in a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    ids = tokenizer.encode(args.sample)
    if hasattr(tokenizer, "encode_to_pieces"):
        pieces = tokenizer.encode_to_pieces(args.sample)
    else:
        pieces = list(args.sample)

    print(f"tokenizer class: {tokenizer.__class__.__name__}")
    print(f"vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "merge_count"):
        print(f"subword merges: {tokenizer.merge_count}")
    print(f"token count: {len(ids)}")
    print("pieces:")
    for token_id, piece in zip(ids, pieces):
        print(f"{token_id:>7}: {piece!r}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
