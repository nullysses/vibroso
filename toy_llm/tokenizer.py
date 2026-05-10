from __future__ import annotations

import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Protocol


_CHUNK_RE = re.compile(r"\s+\S*|[^\s]+")
_SENTENCEPIECE_IMPORT_ERROR = (
    "SentencePiece tokenizer requested, but sentencepiece is not installed. "
    "Install it with: pip install sentencepiece"
)


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
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        token_bytes = {idx: bytes([idx]) for idx in range(256)}
        for left, right in self.merges:
            if left not in token_bytes or right not in token_bytes:
                raise ValueError("Tokenizer merge references an unknown token id")
            token_bytes[len(token_bytes)] = token_bytes[left] + token_bytes[right]

        self.itos = token_bytes
        self.tokens = [self._display_token(self.itos[idx]) for idx in range(len(self.itos))]
        self.chars = self.tokens[:256]
        self._piece_cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()
        self._piece_cache_limit = 50_000
        self._max_chunk_chars = 256

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        max_train_chars: int | None = 10_000,
    ) -> "SubwordTokenizer":
        if not text:
            raise ValueError("Cannot build tokenizer from empty text")
        vocab_size = max(vocab_size, 256)

        train_text = text if max_train_chars is None else text[:max_train_chars]
        chunks = [list(chunk.encode("utf-8")) for chunk in cls._iter_text_chunks(train_text)]
        merges: list[tuple[int, int]] = []
        token_bytes = {idx: bytes([idx]) for idx in range(256)}
        while len(token_bytes) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for chunk in chunks:
                pair_counts.update(zip(chunk, chunk[1:]))
            if not pair_counts:
                break
            pair, count = max(pair_counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
            if count < 2:
                break
            new_token = len(token_bytes)
            token_bytes[new_token] = token_bytes[pair[0]] + token_bytes[pair[1]]
            chunks = [cls._replace_pair(chunk, pair, new_token) for chunk in chunks]
            merges.append(pair)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def merge_count(self) -> int:
        return len(self.merges)

    @staticmethod
    def _display_token(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.hex(" ")

    @staticmethod
    def _replace_pair(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        new_tokens: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    @staticmethod
    def _iter_text_chunks(text: str):
        for match in _CHUNK_RE.finditer(text):
            yield match.group(0)

    def _encode_chunk_to_ids(self, chunk: bytes) -> tuple[int, ...]:
        cached = self._piece_cache.get(chunk)
        if cached is not None:
            self._piece_cache.move_to_end(chunk)
            return cached

        ids = list(chunk)
        for rank, pair in enumerate(self.merges):
            new_token = 256 + rank
            ids = self._replace_pair(ids, pair, new_token)
        encoded = tuple(ids)
        self._piece_cache[chunk] = encoded
        if len(self._piece_cache) > self._piece_cache_limit:
            self._piece_cache.popitem(last=False)
        return encoded

    def _iter_chunks(self, text: str):
        for chunk in self._iter_text_chunks(text):
            for start in range(0, len(chunk), self._max_chunk_chars):
                yield chunk[start : start + self._max_chunk_chars].encode("utf-8")

    def encode_to_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for chunk in self._iter_chunks(text):
            pieces.extend(self._display_token(self.itos[token_id]) for token_id in self._encode_chunk_to_ids(chunk))
        return pieces

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._iter_chunks(text):
            ids.extend(self._encode_chunk_to_ids(chunk))
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        pieces: list[bytes] = []
        for token_id in ids:
            idx = int(token_id)
            try:
                pieces.append(self.itos[idx])
            except KeyError as exc:
                raise ValueError(f"Token id {idx} is not in tokenizer vocabulary") from exc
        try:
            return b"".join(pieces).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Token ids do not decode to valid UTF-8") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "subword",
            "algorithm": "byte_bpe",
            "version": 4,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SubwordTokenizer":
        raw_merges = data.get("merges", [])
        merges = [(int(left), int(right)) for left, right in raw_merges]  # type: ignore[misc]
        return cls(merges)


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"SentencePiece model file not found: {model_file}\n"
                "This checkpoint requires the tokenizer model used during training."
            )
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(_SENTENCEPIECE_IMPORT_ERROR) from exc

        self.model_path = str(model_file)
        self.processor = spm.SentencePieceProcessor()
        loaded = self.processor.Load(self.model_path)
        if not loaded:
            raise ValueError(f"Failed to load SentencePiece model: {self.model_path}")

    @property
    def vocab_size(self) -> int:
        return int(self.processor.GetPieceSize())

    def encode(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.processor.EncodeAsIds(text)]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return str(self.processor.DecodeIds([int(token_id) for token_id in ids]))

    def encode_to_pieces(self, text: str) -> list[str]:
        return [str(piece) for piece in self.processor.EncodeAsPieces(text)]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "sentencepiece",
            "algorithm": "sentencepiece",
            "version": 1,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SentencePieceTokenizer":
        model_path = data.get("model_path")
        if not model_path:
            raise ValueError("SentencePiece tokenizer checkpoint is missing 'model_path'")
        return cls(str(model_path))


def train_sentencepiece_tokenizer(
    text: str,
    model_prefix: str | Path,
    vocab_size: int,
    model_type: str = "bpe",
) -> SentencePieceTokenizer:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError(_SENTENCEPIECE_IMPORT_ERROR) from exc

    if model_type not in {"bpe", "unigram"}:
        raise ValueError("tokenizer_model_type must be 'bpe' or 'unigram'")
    if not text:
        raise ValueError("Cannot build tokenizer from empty text")

    prefix_path = Path(model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    training_path = prefix_path.with_suffix(".train.txt")
    training_path.write_text(text, encoding="utf-8")

    spm.SentencePieceTrainer.Train(
        input=str(training_path),
        model_prefix=str(prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        hard_vocab_limit=False,
    )
    return SentencePieceTokenizer(prefix_path.with_suffix(".model"))


def build_tokenizer(
    text: str,
    kind: str = "subword",
    vocab_size: int = 256,
    max_train_chars: int | None = 10_000,
    sentencepiece_model_path: str | None = None,
    sentencepiece_prefix: str | None = None,
    sentencepiece_model_type: str = "bpe",
) -> Tokenizer:
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "subword":
        return SubwordTokenizer.from_text(
            text,
            vocab_size=vocab_size,
            max_train_chars=max_train_chars,
        )
    if kind == "sentencepiece":
        if sentencepiece_model_path and Path(sentencepiece_model_path).exists():
            return SentencePieceTokenizer(sentencepiece_model_path)
        if not sentencepiece_prefix:
            raise ValueError(
                "tokenizer_prefix must be provided when training a SentencePiece tokenizer"
            )
        train_text = text if max_train_chars is None else text[:max_train_chars]
        return train_sentencepiece_tokenizer(
            train_text,
            model_prefix=sentencepiece_prefix,
            vocab_size=vocab_size,
            model_type=sentencepiece_model_type,
        )
    raise ValueError("Tokenizer kind must be 'subword', 'char', or 'sentencepiece'")


def tokenizer_from_dict(data: dict[str, object]) -> Tokenizer:
    kind = data.get("kind")
    if kind == "subword":
        if data.get("algorithm") != "byte_bpe" or int(data.get("version", 1)) < 4:
            raise ValueError(
                "This subword tokenizer checkpoint is incompatible with the current "
                "byte-level BPE tokenizer. Retrain from scratch to create a new checkpoint."
            )
        return SubwordTokenizer.from_dict(data)
    if kind == "sentencepiece":
        if data.get("algorithm") != "sentencepiece" or int(data.get("version", 0)) < 1:
            raise ValueError("SentencePiece tokenizer checkpoint is incompatible")
        return SentencePieceTokenizer.from_dict(data)
    if kind == "char" or kind is None:
        return CharTokenizer.from_dict(data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown tokenizer kind in checkpoint: {kind!r}")
