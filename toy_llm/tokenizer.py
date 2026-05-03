from __future__ import annotations


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

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        if "chars" not in data:
            raise ValueError("Tokenizer data must contain 'chars'")
        return cls(list(data["chars"]))
