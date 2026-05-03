from __future__ import annotations

from collections import Counter
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        ...

    def to_dict(self) -> dict:
        ...


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if not chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        if len(set(chars)) != len(chars):
            raise ValueError("Tokenizer vocabulary contains duplicate characters")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot build tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            try:
                ids.append(self.stoi[ch])
            except KeyError as exc:
                raise ValueError(f"Character {ch!r} is not in tokenizer vocabulary") from exc
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        chars: list[str] = []
        for token_id in ids:
            idx = int(token_id)
            try:
                chars.append(self.itos[idx])
            except KeyError as exc:
                raise ValueError(f"Token id {idx} is not in tokenizer vocabulary") from exc
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        return {"kind": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        if "chars" not in data:
            raise ValueError("Tokenizer data must contain 'chars'")
        return cls(list(data["chars"]))


class SubwordTokenizer:
    def __init__(self, chars: list[str], merges: list[tuple[str, str]]):
        if not chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        if len(set(chars)) != len(chars):
            raise ValueError("Tokenizer vocabulary contains duplicate base characters")
        self.chars = list(chars)
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        vocab = list(self.chars)
        seen = set(vocab)
        for left, right in self.merges:
            token = left + right
            if token not in seen:
                vocab.append(token)
                seen.add(token)
        self.tokens = vocab
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}
        self.base_chars = set(self.chars)

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 256) -> "SubwordTokenizer":
        if not text:
            raise ValueError("Cannot build tokenizer from empty text")
        chars = sorted(set(text))
        if vocab_size < len(chars):
            raise ValueError(
                f"tokenizer_vocab_size {vocab_size} is smaller than the base character "
                f"vocabulary size {len(chars)}"
            )

        tokens = list(text)
        merges: list[tuple[str, str]] = []
        known_tokens = set(chars)
        while len(known_tokens) < vocab_size:
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break
            pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            merged = pair[0] + pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            merges.append(pair)
            known_tokens.add(merged)
        return cls(chars, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        missing = sorted({ch for ch in text if ch not in self.base_chars})
        if missing:
            raise ValueError(f"Characters are not in tokenizer vocabulary: {missing!r}")

        tokens = list(text)
        for pair in self.merges:
            merged = pair[0] + pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.stoi[token] for token in tokens]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            idx = int(token_id)
            try:
                pieces.append(self.itos[idx])
            except KeyError as exc:
                raise ValueError(f"Token id {idx} is not in tokenizer vocabulary") from exc
        return "".join(pieces)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "subword",
            "chars": self.chars,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SubwordTokenizer":
        if "chars" not in data:
            raise ValueError("Tokenizer data must contain 'chars'")
        raw_merges = data.get("merges", [])
        merges = [(str(left), str(right)) for left, right in raw_merges]  # type: ignore[misc]
        return cls(list(data["chars"]), merges)  # type: ignore[arg-type]


def build_tokenizer(text: str, kind: str = "subword", vocab_size: int = 256) -> Tokenizer:
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "subword":
        return SubwordTokenizer.from_text(text, vocab_size=vocab_size)
    raise ValueError("Tokenizer kind must be 'subword' or 'char'")


def tokenizer_from_dict(data: dict[str, object]) -> Tokenizer:
    kind = data.get("kind")
    if kind == "subword":
        return SubwordTokenizer.from_dict(data)
    if kind == "char" or kind is None:
        return CharTokenizer.from_dict(data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown tokenizer kind in checkpoint: {kind!r}")
