from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.checkpoint import load_checkpoint
from toy_llm.config import TrainConfig
from toy_llm.device import get_device
from toy_llm.model import TinyGPT
from toy_llm.tokenizer import tokenizer_from_dict


DEFAULT_PROMPTS = [
    "Tortilla\nA tortilla is",
    "Artificial intelligence\nArtificial intelligence is",
    "Physics\nPhysics is",
    "Music\nMusic is",
    "Mexico\nMexico is",
    "The Roman Empire\nThe Roman Empire was",
]


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        default_path = ROOT / "benchmarks" / "prompts.txt"
        if default_path.exists():
            path = str(default_path)
        else:
            return DEFAULT_PROMPTS

    text = Path(path).read_text(encoding="utf-8")
    prompts = [block.strip("\n") for block in text.split("---") if block.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Vibroso generation.")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
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

    sections = [
        "=== Generation Benchmark ===",
        f"checkpoint: {args.checkpoint}",
        f"temperature: {args.temperature}",
        f"top_k: {args.top_k}",
        f"max_new_tokens: {args.max_new_tokens}",
        f"seed: {args.seed}",
        "",
    ]

    for prompt in load_prompts(args.prompt_file):
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        sections.extend(
            [
                "=" * 80,
                f"PROMPT: {prompt!r}",
                "-" * 80,
                tokenizer.decode(out[0].tolist()),
                "",
            ]
        )

    output_text = "\n".join(sections)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
