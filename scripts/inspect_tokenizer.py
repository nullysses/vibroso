from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.config import Config
from toy_llm.dataset import load_corpus
from toy_llm.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the character tokenizer for a dataset.")
    parser.add_argument("--dataset", default="data/input.txt")
    parser.add_argument("--url-list", action="store_true", help="Treat --dataset as a URL list.")
    parser.add_argument("--cache", default=None, help="Optional cache file for fetched URL corpus.")
    args = parser.parse_args()

    config = Config(
        dataset_path=args.dataset,
        dataset_kind="url_list" if args.url_list else "text",
        corpus_cache_path=args.cache,
    )
    text = load_corpus(config)
    tokenizer = CharTokenizer.from_text(text)
    print(f"characters: {len(text)}")
    print(f"tokens: {len(tokenizer.encode(text))}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print("vocabulary:")
    for idx, ch in enumerate(tokenizer.chars):
        printable = ch if ch not in {"\n", "\t", "\r"} else repr(ch)
        print(f"{idx:>3}: {printable}")


if __name__ == "__main__":
    main()
