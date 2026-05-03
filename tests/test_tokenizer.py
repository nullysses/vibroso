import pytest

from toy_llm.tokenizer import CharTokenizer


def test_vocabulary_construction_is_deterministic():
    tokenizer = CharTokenizer.from_text("banana")
    assert tokenizer.chars == ["a", "b", "n"]
    assert tokenizer.vocab_size == 3


def test_encode_decode_round_trip():
    tokenizer = CharTokenizer.from_text("hello")
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_unknown_character_raises_clear_error():
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in tokenizer vocabulary"):
        tokenizer.encode("abd")


def test_serialized_tokenizer_can_be_restored():
    tokenizer = CharTokenizer.from_text("cab")
    restored = CharTokenizer.from_dict(tokenizer.to_dict())
    assert restored.chars == ["a", "b", "c"]
    assert restored.decode(restored.encode("cab")) == "cab"
