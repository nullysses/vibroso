import pytest

from toy_llm.config import Config, InferenceConfig, ModelConfig, TrainConfig


def test_config_alias_points_to_train_config():
    assert Config is TrainConfig


def test_train_config_validates_model_shape():
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig.from_dict(
            {
                "block_size": 8,
                "model": {"n_embd": 10, "n_head": 3, "n_layer": 1, "dropout": 0.0},
            }
        )

    with pytest.raises(ValueError, match="must be even for RoPE"):
        TrainConfig.from_dict(
            {
                "block_size": 8,
                "model": {"n_embd": 12, "n_head": 4, "n_layer": 1, "dropout": 0.0},
            }
        )


def test_model_config_rejects_unknown_mlp_type():
    with pytest.raises(ValueError, match="model.mlp_type"):
        ModelConfig(mlp_type="unknown").validate()


def test_model_config_accepts_swiglu_mlp_type():
    config = ModelConfig(mlp_type="swiglu")
    config.validate()
    assert config.mlp_type == "swiglu"


def test_train_config_accepts_instruction_jsonl_dataset_kind():
    config = TrainConfig.from_dict(
        {
            "dataset_path": "data/instruction_examples.example.jsonl",
            "dataset_kind": "instruction_jsonl",
        }
    )
    assert config.dataset_kind == "instruction_jsonl"


def test_train_config_accepts_sentencepiece_tokenizer_fields():
    config = TrainConfig.from_dict(
        {
            "tokenizer_kind": "sentencepiece",
            "tokenizer_vocab_size": 1024,
            "tokenizer_model_type": "bpe",
            "tokenizer_prefix": "tokenizers/sentencepiece/test_1024",
            "tokenizer_model_path": "tokenizers/sentencepiece/test_1024.model",
        }
    )
    assert config.tokenizer_kind == "sentencepiece"
    assert config.tokenizer_model_type == "bpe"


def test_train_config_rejects_invalid_sentencepiece_settings():
    with pytest.raises(ValueError, match="tokenizer_model_type"):
        TrainConfig.from_dict(
            {
                "tokenizer_kind": "sentencepiece",
                "tokenizer_model_type": "word",
                "tokenizer_prefix": "tokenizers/sentencepiece/test",
            }
        )

    with pytest.raises(ValueError, match="tokenizer_model_path or tokenizer_prefix"):
        TrainConfig.from_dict({"tokenizer_kind": "sentencepiece"})


def test_inference_config_validates_sampling_values():
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig(prompt="hello", temperature=0).validate()
    with pytest.raises(ValueError, match="top_k"):
        InferenceConfig(prompt="hello", top_k=0).validate()


def test_inference_config_cli_style_overrides():
    config = InferenceConfig(
        checkpoint="checkpoints/old.pt",
        prompt="hello",
        max_new_tokens=20,
        temperature=0.8,
        top_k=10,
        device="cpu",
    )
    updated = config.with_overrides(prompt="world", max_new_tokens=5, top_k=None)
    assert updated.checkpoint == "checkpoints/old.pt"
    assert updated.prompt == "world"
    assert updated.max_new_tokens == 5
    assert updated.top_k == 10
