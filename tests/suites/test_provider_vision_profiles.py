"""Tests for provider/model vision-output profile helpers."""

import pytest

from face_of_agi.models.providers.vision import resolve_model_vision_profile


def test_known_model_profiles_use_normalized_1000_coordinates() -> None:
    profiles = [
        resolve_model_vision_profile(backend="openai", model="gpt-5-nano"),
        resolve_model_vision_profile(backend="openai", model="gpt-5.5"),
        resolve_model_vision_profile(backend="ollama", model="gemma4:e4b"),
        resolve_model_vision_profile(backend="ollama", model="gemma4:26b"),
        resolve_model_vision_profile(backend="ollama", model="qwen3.6:35b"),
        resolve_model_vision_profile(
            backend="vllm",
            model="Qwen/Qwen3.6-35B-A3B-FP8",
        ),
    ]

    assert [profile.coordinate_space for profile in profiles] == [
        "normalized_1000",
        "normalized_1000",
        "normalized_1000",
        "normalized_1000",
        "normalized_1000",
        "normalized_1000",
    ]
    assert {profile.source for profile in profiles} == {"model_profile"}


def test_unknown_model_requires_profile_entry() -> None:
    with pytest.raises(ValueError, match="add it to vision_profiles.json"):
        resolve_model_vision_profile(
            backend="ollama",
            model="some-new-vlm:latest",
        )
