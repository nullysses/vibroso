import pytest

from toy_llm.tokenizer import CharTokenizer, SubwordTokenizer, build_tokenizer, tokenizer_from_dict


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


def test_subword_tokenizer_round_trips_and_learns_merges():
    tokenizer = SubwordTokenizer.from_text("abababab", vocab_size=258)
    assert tokenizer.decode(tokenizer.encode("abab")) == "abab"
    assert tokenizer.vocab_size > 256
    assert len(tokenizer.encode("abab")) < 4
    assert tokenizer.encode_to_pieces("abab") != list("abab")


def test_subword_tokenizer_serializes_and_restores():
    tokenizer = SubwordTokenizer.from_text("hello hello hello", vocab_size=270)
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert restored.decode(restored.encode("hello hello")) == "hello hello"
    assert restored.vocab_size == tokenizer.vocab_size


def test_build_tokenizer_defaults_to_subword():
    tokenizer = build_tokenizer("abc abc abc", vocab_size=260)
    assert isinstance(tokenizer, SubwordTokenizer)


def test_subword_tokenizer_expands_to_fit_base_bytes():
    tokenizer = SubwordTokenizer.from_text("abcd", vocab_size=2)
    assert tokenizer.vocab_size == 256
    assert tokenizer.decode(tokenizer.encode("abcd")) == "abcd"


def test_subword_tokenizer_shortens_repeated_sentence():
    sentence = "A tortilla is a traditional Mexican flatbread."
    corpus = (sentence + "\n") * 20
    tokenizer = SubwordTokenizer.from_text(corpus, vocab_size=320)
    pieces = tokenizer.encode_to_pieces(sentence)
    assert tokenizer.decode(tokenizer.encode(sentence)) == sentence
    assert len(pieces) < len(sentence)
    assert any(len(piece) > 1 for piece in pieces)


def test_subword_training_window_still_keeps_full_base_vocabulary():
    tokenizer = SubwordTokenizer.from_text("aaaa xyz", vocab_size=6, max_train_chars=4)
    assert tokenizer.decode(tokenizer.encode("xyz 😄")) == "xyz 😄"


def test_subword_encoder_caches_repeated_chunks():
    tokenizer = SubwordTokenizer.from_text("hello world " * 20, vocab_size=290)
    text = "hello world " * 10
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(tokenizer._piece_cache) < 5


def test_subword_tokenizer_handles_unseen_unicode_bytes():
    tokenizer = SubwordTokenizer.from_text("plain ascii training", vocab_size=260)
    text = "jalapeno 😄"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_old_subword_checkpoint_payload_is_rejected():
    payload = {
        "kind": "subword",
        "version": 2,
        "chars": ["a", "b"],
        "merges": [["a", "b"]],
    }
    with pytest.raises(ValueError, match="incompatible"):
        tokenizer_from_dict(payload)
