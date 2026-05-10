from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_llm.config import TrainConfig
from toy_llm.dataset import load_corpus
from toy_llm.tokenizer import build_tokenizer


SAMPLES = [
    "A tortilla is a traditional Mexican flatbread.",
    "Artificial intelligence is a field of computer science.",
    "The Roman Empire was one of the largest empires in history.",
    "Machine learning models are trained using data.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Vibroso tokenizer compression.")
    parser.add_argument("--config", default="configs/links.yaml", help="Training config path.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)
    text = load_corpus(config)
    tokenizer = build_tokenizer(
        text,
        kind=config.tokenizer_kind,
        vocab_size=config.tokenizer_vocab_size,
        max_train_chars=config.tokenizer_train_chars,
        sentencepiece_model_path=config.tokenizer_model_path,
        sentencepiece_prefix=config.tokenizer_prefix,
        sentencepiece_model_type=config.tokenizer_model_type,
    )

    ids = tokenizer.encode(text)
    dataset_tokens = len(ids)
    dataset_bytes = len(text.encode("utf-8"))
    chars_per_token = len(text) / max(dataset_tokens, 1)
    bytes_per_token = dataset_bytes / max(dataset_tokens, 1)
    merge_count = getattr(tokenizer, "merge_count", None)

    print("=== Tokenizer Benchmark ===")
    print(f"tokenizer kind: {config.tokenizer_kind}")
    print(f"vocab size: {tokenizer.vocab_size}")
    if merge_count is not None:
        print(f"merge count: {merge_count}")
    print(f"dataset chars: {len(text):,}")
    print(f"dataset tokens: {dataset_tokens:,}")
    print(f"chars/token: {chars_per_token:.3f}")
    print(f"bytes/token: {bytes_per_token:.3f}")

    sample_records: list[dict[str, object]] = []
    print()
    print("=== Sample Encodings ===")
    for sample in SAMPLES:
        sample_ids = tokenizer.encode(sample)
        pieces = tokenizer.encode_to_pieces(sample) if hasattr(tokenizer, "encode_to_pieces") else list(sample)
        sample_record = {
            "sample": sample,
            "tokens": len(sample_ids),
            "chars_per_token": len(sample) / max(len(sample_ids), 1),
            "pieces": pieces,
        }
        sample_records.append(sample_record)
        print()
        print(f"sample: {sample!r}")
        print(f"tokens: {len(sample_ids)}")
        print(f"chars/token: {sample_record['chars_per_token']:.3f}")
        print("pieces:")
        print(pieces)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_kind": config.tokenizer_kind,
            "vocab_size": tokenizer.vocab_size,
            "merge_count": merge_count,
            "dataset_chars": len(text),
            "dataset_bytes": dataset_bytes,
            "dataset_tokens": dataset_tokens,
            "chars_per_token": chars_per_token,
            "bytes_per_token": bytes_per_token,
            "samples": sample_records,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
