import pytest

from toy_llm.config import Config, InferenceConfig, TrainConfig


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
