"""Runtime/config smoke tests for the online learner path."""

from __future__ import annotations

from pathlib import Path

import pytest

from face_of_agi.environment.config import (
    AgentRuntimeConfig,
    BackboneRuntimeConfig,
    OnlineRuntimeConfig,
    load_environment_config,
)
from face_of_agi.online.factory import build_online_agent


def test_environment_config_loads_online_agent_shape(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(_active_config_yaml(), encoding="utf-8")

    config = load_environment_config(path)

    assert config.agent.backbone.backend == "transformers"
    assert config.agent.backbone.model_family == "qwen3_5_moe_multimodal"
    assert (
        config.agent.backbone.model_path
        == "/kaggle/input/face-of-agi-qwen36-35b-fp8-weights"
    )
    assert config.agent.backbone.local_files_only is True
    assert config.agent.backbone.representation_layer == "image_tokens_mean"
    assert config.agent.backbone.model_kwargs == {"device_map": "auto"}
    assert config.agent.online.buffer_size == 512
    assert config.agent.replay.max_seconds_per_turn == 0.5
    assert config.agent.planner.coordinate_candidates == 16
    assert config.animation_keyframe_pixel_threshold == 8


def test_environment_config_rejects_removed_models_config(tmp_path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "game_index: 0\nmax_actions_per_level: 1\nmodels: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models config has been removed"):
        load_environment_config(path)


def test_environment_config_rejects_missing_backbone_model_path(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "agent:\n"
        "  backbone:\n"
        "    backend: transformers\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agent.backbone.model_path is required"):
        load_environment_config(path)


def test_environment_config_rejects_non_transformers_yaml_backend(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "agent:\n"
        "  backbone:\n"
        "    backend: vllm\n"
        "    model_path: /kaggle/input/model\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agent.backbone.backend must be transformers"):
        load_environment_config(path)


def test_environment_config_rejects_negative_animation_keyframe_threshold(
    tmp_path,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + "animation_keyframe_pixel_threshold: -1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="animation_keyframe_pixel_threshold"):
        load_environment_config(path)


def test_committed_runtime_configs_load_online_agent_without_removed_keys() -> None:
    config_root = Path("src/face_of_agi/runtime/configs")
    config_paths = sorted(config_root.rglob("*.yaml"))

    assert config_paths
    for path in config_paths:
        config = load_environment_config(path)
        assert config.agent.backbone.backend == "transformers"
        assert config.agent.backbone.model_family == "qwen3_5_moe_multimodal"
        assert config.agent.backbone.model_path
        assert config.agent.replay.max_updates_per_turn >= 0


def test_build_online_agent_wires_fake_backbone_for_tests() -> None:
    config = AgentRuntimeConfig(
        backbone=BackboneRuntimeConfig(
            backend="deterministic",
            model_path="unused-test-backbone",
        ),
        online=OnlineRuntimeConfig(hidden_dim=8, ensemble_size=2),
    )

    agent = build_online_agent(config)

    assert agent.snapshot()["backbone"] == {
        "backend": "deterministic",
        "feature_dim": 8,
    }


def _active_config_yaml() -> str:
    return (
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "operation_mode: offline\n"
        "agent:\n"
        "  backbone:\n"
        "    backend: transformers\n"
        "    model_family: qwen3_5_moe_multimodal\n"
        "    model_path: /kaggle/input/face-of-agi-qwen36-35b-fp8-weights\n"
        "    processor_path: null\n"
        "    device: auto\n"
        "    dtype: auto\n"
        "    image_size: 224x224\n"
        "    local_files_only: true\n"
        "    representation_layer: image_tokens_mean\n"
        "    model_kwargs:\n"
        "      device_map: auto\n"
        "  online:\n"
        "    buffer_size: 512\n"
        "    adapter_rank: 16\n"
        "    ensemble_size: 5\n"
        "    hidden_dim: 512\n"
        "    learning_rate: 0.001\n"
        "    batch_size: 32\n"
        "  replay:\n"
        "    max_updates_per_turn: 8\n"
        "    max_seconds_per_turn: 0.5\n"
        "    solved_level_updates: 32\n"
        "  planner:\n"
        "    horizon: 3\n"
        "    candidate_count: 64\n"
        "    coordinate_candidates: 16\n"
        "    diagnostic_turns: 4\n"
    )
