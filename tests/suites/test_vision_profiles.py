"""Tests for provider/model visual coordinate profiles."""

from __future__ import annotations

from face_of_agi.models.providers.vision import resolve_model_vision_profile


def test_resolves_gemma4_fp8_vllm_profile() -> None:
    profile = resolve_model_vision_profile(
        backend="vllm",
        model="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
    )

    assert profile.input_image_size == (1810, 1810)
    assert profile.coordinate_space == "normalized_1000"
    assert profile.bbox_order == "xyxy"
    assert profile.axis_frame == "top_left_x_right_y_down"


def test_vision_profile_lookup_normalizes_case() -> None:
    profile = resolve_model_vision_profile(
        backend="VLLM",
        model="redhatai/gemma-4-26b-a4b-it-fp8-dynamic",
    )

    assert profile.coordinate_space == "normalized_1000"


def test_resolves_minicpm_v46_thinking_vllm_profile() -> None:
    profile = resolve_model_vision_profile(
        backend="vllm",
        model="openbmb/MiniCPM-V-4.6-Thinking",
    )

    assert profile.input_image_size == (2048, 2048)
    assert profile.coordinate_space == "normalized_1000"
    assert profile.bbox_order == "xyxy"
    assert profile.axis_frame == "top_left_x_right_y_down"
