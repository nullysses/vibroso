from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.dataset import format_instruction_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview formatted instruction JSONL examples.")
    parser.add_argument("--dataset", default="data/instruction_examples.example.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    path = Path(args.dataset)
    shown = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid instruction JSONL row at line {line_no}: expected object")
            print(f"=== Example {shown + 1} ===")
            print(
                format_instruction_example(
                    instruction=str(row.get("instruction", "")),
                    response=str(row.get("response", "")),
                    context=row.get("context") if isinstance(row.get("context"), str) else None,
                ),
                end="",
            )
            shown += 1
            if shown >= args.limit:
                break
    if shown == 0:
        raise ValueError(f"No examples found in {path}")


if __name__ == "__main__":
    main()
