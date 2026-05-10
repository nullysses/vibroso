from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.config import TrainConfig
from toy_llm.dataset import load_corpus
from toy_llm.tokenizer import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the character tokenizer for a dataset.")
    parser.add_argument("--dataset", default="data/input.txt")
    parser.add_argument("--url-list", action="store_true", help="Treat --dataset as a URL list.")
    parser.add_argument(
        "--wikipedia-titles",
        action="store_true",
        help="Treat --dataset as a list of Wikipedia page titles.",
    )
    parser.add_argument("--cache", default=None, help="Optional cache file for fetched URL corpus.")
    parser.add_argument("--language", default="en", help="Wikipedia language code.")
    parser.add_argument("--tokenizer", choices=["subword", "char", "sentencepiece"], default="subword")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--tokenizer-train-chars", type=int, default=10_000)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-prefix", default=None)
    parser.add_argument("--tokenizer-model-type", choices=["bpe", "unigram"], default="bpe")
    args = parser.parse_args()

    dataset_kind = "text"
    if args.url_list:
        dataset_kind = "url_list"
    if args.wikipedia_titles:
        dataset_kind = "wikipedia_titles"

    config = TrainConfig(
        dataset_path=args.dataset,
        dataset_kind=dataset_kind,
        corpus_cache_path=args.cache,
        wikipedia_language=args.language,
    )
    text = load_corpus(config)
    tokenizer = build_tokenizer(
        text,
        kind=args.tokenizer,
        vocab_size=args.vocab_size,
        max_train_chars=args.tokenizer_train_chars,
        sentencepiece_model_path=args.tokenizer_model_path,
        sentencepiece_prefix=args.tokenizer_prefix,
        sentencepiece_model_type=args.tokenizer_model_type,
    )
    print(f"characters: {len(text)}")
    print(f"tokens: {len(tokenizer.encode(text))}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "merge_count"):
        print(f"subword merges: {tokenizer.merge_count}")
    tokens = getattr(tokenizer, "tokens", getattr(tokenizer, "chars", None))
    if tokens is not None:
        print("vocabulary:")
        for idx, token in enumerate(tokens):
            printable = token if token not in {"\n", "\t", "\r"} else repr(token)
            print(f"{idx:>3}: {printable}")


if __name__ == "__main__":
    main()
