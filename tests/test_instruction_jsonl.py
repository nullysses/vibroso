import pytest

from toy_llm.config import TrainConfig
from toy_llm.dataset import format_instruction_example, load_corpus, load_instruction_jsonl


def test_format_instruction_example_without_context():
    text = format_instruction_example(" What is a tortilla? ", " A flatbread. ")
    assert text == "<|user|>\nWhat is a tortilla?\n<|assistant|>\nA flatbread.\n<|end|>\n"


def test_load_instruction_jsonl_basic(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text(
        '{"instruction":"What is a tortilla?","response":"A flatbread."}\n',
        encoding="utf-8",
    )

    text = load_instruction_jsonl(path)

    assert "<|user|>" in text
    assert "What is a tortilla?" in text
    assert "<|assistant|>" in text
    assert "A flatbread." in text
    assert "<|end|>" in text


def test_load_instruction_jsonl_with_context(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text(
        (
            '{"instruction":"Summarize.","context":"A tortilla is a flatbread.",'
            '"response":"A tortilla is a flatbread."}\n'
        ),
        encoding="utf-8",
    )

    text = load_instruction_jsonl(path)

    assert "Context:" in text
    assert "A tortilla is a flatbread." in text


def test_load_instruction_jsonl_separates_examples(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"instruction":"One?","response":"First."}',
                '{"instruction":"Two?","response":"Second."}',
            ]
        ),
        encoding="utf-8",
    )

    text = load_instruction_jsonl(path)

    assert "<|end|>\n\n<|user|>" in text


def test_load_instruction_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_instruction_jsonl(path)


def test_load_instruction_jsonl_rejects_non_object_rows(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('["not", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        load_instruction_jsonl(path)


def test_load_instruction_jsonl_rejects_missing_response(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction":"What is a tortilla?"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="field 'response'"):
        load_instruction_jsonl(path)


def test_load_instruction_jsonl_rejects_non_string_context(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"instruction":"Summarize.","context":["bad"],"response":"No."}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field 'context' must be a string"):
        load_instruction_jsonl(path)


def test_load_instruction_jsonl_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no examples"):
        load_instruction_jsonl(path)


def test_load_corpus_routes_instruction_jsonl(tmp_path):
    path = tmp_path / "instructions.jsonl"
    path.write_text('{"instruction":"What is Vibroso?","response":"A toy model."}\n', encoding="utf-8")
    config = TrainConfig(dataset_path=str(path), dataset_kind="instruction_jsonl")

    text = load_corpus(config)

    assert "What is Vibroso?" in text
    assert "A toy model." in text
