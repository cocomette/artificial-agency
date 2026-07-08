"""Runtime/config smoke tests for the active role set."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from face_of_agi.environment.config import (
    ModelRoleConfig,
    load_environment_config,
)
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.world.config import VLLMWorldConfig
from face_of_agi.runtime.shell import _build_model_registry, _config_kwargs


def test_environment_config_loads_active_model_shape(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(_active_config_yaml(), encoding="utf-8")

    config = load_environment_config(path)

    assert config.models.agent.backend == "vllm"
    assert config.models.change.backend == "vllm"
    assert config.models.memory.backend == "vllm"
    assert config.models.world.backend == "vllm"
    assert config.models.goal.backend == "vllm"
    assert config.models.reward_judge.backend == "vllm"
    assert not hasattr(config, "online_lora")
    assert config.animation_keyframe_pixel_threshold == 8


def test_vllm_config_kwargs_include_shared_runtime_options() -> None:
    shared = ModelRoleConfig(
        backend="vllm",
        model="shared-qwen",
        options={
            "base_url": "http://127.0.0.1:8000/v1",
            "temperature": 0.2,
        },
    )
    role = ModelRoleConfig(
        backend="vllm",
        options={"max_completion_tokens": 128},
    )

    from face_of_agi.runtime.shell import _with_shared_vlm_role_config

    merged = _with_shared_vlm_role_config(role, shared)

    world_kwargs = _config_kwargs(merged, VLLMWorldConfig)
    agent_kwargs = _config_kwargs(merged, VLLMOrchestratorAgentConfig)
    assert world_kwargs["model"] == "shared-qwen"
    assert world_kwargs["base_url"] == "http://127.0.0.1:8000/v1"
    assert world_kwargs["temperature"] == 0.2
    assert agent_kwargs["max_completion_tokens"] == 128


def test_environment_config_loads_animation_keyframe_threshold(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + "animation_keyframe_pixel_threshold: 3\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.animation_keyframe_pixel_threshold == 3


def test_environment_config_rejects_online_lora_block(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "online_lora:\n"
        + "  enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="online_lora has been removed"):
        load_environment_config(path)


def test_schema_instruction_flag_lands_on_vllm_config_not_options() -> None:
    role_config = ModelRoleConfig(
        backend="vllm",
        model="model",
        options={
            "include_output_schema_in_instructions": True,
            "temperature": 0.1,
        },
    )

    world = VLLMWorldConfig(**_config_kwargs(role_config, VLLMWorldConfig))
    agent = VLLMOrchestratorAgentConfig(
        **_config_kwargs(role_config, VLLMOrchestratorAgentConfig)
    )

    assert world.include_output_schema_in_instructions is True
    assert agent.include_output_schema_in_instructions is True
    assert "include_output_schema_in_instructions" not in world.options
    assert "include_output_schema_in_instructions" not in agent.options


def test_environment_config_rejects_negative_animation_keyframe_threshold(
    tmp_path,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + "animation_keyframe_pixel_threshold: -1\n",
        encoding="utf-8",
    )

    try:
        load_environment_config(path)
    except ValueError as exc:
        assert "animation_keyframe_pixel_threshold must be non-negative" in str(exc)
    else:
        raise AssertionError("expected config validation to reject negative threshold")


def test_committed_runtime_configs_load_without_removed_keys() -> None:
    config_root = "src/face_of_agi/runtime/configs"
    output = subprocess.check_output(
        ["git", "ls-files", config_root],
        text=True,
    )
    config_paths = sorted(
        Path(line)
        for line in output.splitlines()
        if line.endswith(".yaml") and Path(line).exists()
    )

    assert config_paths
    for path in config_paths:
        config = load_environment_config(path)
        assert not hasattr(config, "online_lora")


def test_environment_config_rejects_removed_role_keys(tmp_path) -> None:
    path = tmp_path / "removed-role.yaml"
    path.write_text(
        _active_config_yaml()
        + "  historizer:\n"
        + "    backend: vllm\n"
        + "    model: qwen\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.historizer"):
        load_environment_config(path)


def test_environment_config_rejects_removed_normalized_crop_key(tmp_path) -> None:
    path = tmp_path / "removed-crop.yaml"
    path.write_text(
        _active_config_yaml()
        + "  agent:\n"
        + "    input_image_crop_box_normalized: [0, 0, 1, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input_image_crop_arc_grid_edges"):
        load_environment_config(path)


def test_build_model_registry_wires_active_vllm_roles_only() -> None:
    model = "Qwen/Qwen3.6-35B-A3B-FP8"
    role = ModelRoleConfig(backend="vllm", model=model)

    registry = _build_model_registry(
        agent_config=role,
        change_config=role,
        memory_config=role,
        world_config=role,
        goal_config=role,
        interest_config=role,
        reward_judge_config=role,
        shared_vlm_config=ModelRoleConfig(),
    )

    assert registry.orchestrator_agent is not None
    assert registry.change_summary_model is not None
    assert registry.memory_model is not None
    assert registry.world_model is not None
    assert registry.goal_model is not None
    assert registry.interest_model is not None
    assert registry.reward_judge_model is not None


def _active_config_yaml(*, shared_server_model_path: str | None = None) -> str:
    shared = ""
    if shared_server_model_path is not None:
        shared = (
            "  shared_vlm:\n"
            "    backend: vllm\n"
            "    model: shared-qwen\n"
            "    server:\n"
            f"      model_path: {shared_server_model_path}\n"
        )
    return (
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "operation_mode: offline\n"
        "models:\n"
        + shared +
        "  agent:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  change:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  memory:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  world:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  goal:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  interest:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  reward_judge:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
    )
