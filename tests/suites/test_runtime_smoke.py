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
from face_of_agi.models.orchestrator_agent.config import HFOrchestratorAgentConfig
from face_of_agi.models.world.config import HFWorldConfig
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
    assert config.online_lora.update_interval_turns == 4
    assert config.animation_keyframe_pixel_threshold == 8


def test_environment_config_loads_hf_transformers_debug_config() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/hf/hf_h100_qwen36_35b_bnb4_debug.yaml"
    )

    assert config.models.shared_vlm.backend == "hf_transformers"
    assert config.models.shared_vlm.model == "Qwen/Qwen3.6-35B-A3B"
    assert config.models.shared_vlm.options["quantization"] == "bnb_4bit"
    assert config.models.shared_vlm.options["local_files_only"] is True
    assert config.models.agent.backend == "hf_transformers"
    assert config.models.memory.backend == "hf_transformers"
    assert config.models.world.backend == "hf_transformers"
    assert config.online_lora.base_model == "Qwen/Qwen3.6-35B-A3B"
    assert config.online_lora.trainer_base_model == "Qwen/Qwen3.6-35B-A3B"


def test_hf_config_kwargs_include_shared_runtime_options() -> None:
    shared = ModelRoleConfig(
        backend="hf_transformers",
        model="trainable-qwen",
        options={
            "local_files_only": True,
            "quantization": "bnb_4bit",
            "device_map": "cuda:0",
            "lora_target_modules": ["q_proj", "v_proj"],
        },
    )
    role = ModelRoleConfig(
        backend="hf_transformers",
        options={"max_completion_tokens": 128},
    )

    from face_of_agi.runtime.shell import _with_shared_vlm_role_config

    merged = _with_shared_vlm_role_config(role, shared)

    world_kwargs = _config_kwargs(merged, HFWorldConfig)
    agent_kwargs = _config_kwargs(merged, HFOrchestratorAgentConfig)
    assert world_kwargs["model"] == "trainable-qwen"
    assert world_kwargs["quantization"] == "bnb_4bit"
    assert world_kwargs["local_files_only"] is True
    assert world_kwargs["lora_target_modules"] == ["q_proj", "v_proj"]
    assert agent_kwargs["max_completion_tokens"] == 128


def test_environment_config_loads_animation_keyframe_threshold(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + "animation_keyframe_pixel_threshold: 3\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.animation_keyframe_pixel_threshold == 3


def test_environment_config_loads_online_lora_grpo_controls(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "online_lora:\n"
        + "  train_batch_size: 3\n"
        + "  train_epochs: 2\n"
        + "  max_update_steps: 5\n"
        + "  max_concurrent_trainer_jobs: 2\n"
        + "  max_update_wait_seconds: 12.5\n"
        + "  trainer_cache_enabled: false\n"
        + "  trainer_base_model: trainable-qwen\n"
        + "  trainer_base_model_path: /models/trainable-qwen\n"
        + "  trainer_local_files_only: true\n"
        + "  trainer_quantization: bnb_4bit\n"
        + "  trainer_device_map: none\n"
        + "  trainer_torch_dtype: bf16\n"
        + "  lora_target_modules: [q_proj, v_proj]\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.online_lora.train_batch_size == 3
    assert config.online_lora.train_epochs == 2
    assert config.online_lora.max_update_steps == 5
    assert config.online_lora.max_concurrent_trainer_jobs == 2
    assert config.online_lora.max_update_wait_seconds == 12.5
    assert config.online_lora.trainer_cache_enabled is False
    assert config.online_lora.trainer_base_model == "trainable-qwen"
    assert config.online_lora.trainer_base_model_path == "/models/trainable-qwen"
    assert config.online_lora.trainer_local_files_only is True
    assert config.online_lora.trainer_quantization == "bnb_4bit"
    assert config.online_lora.trainer_device_map == "none"
    assert config.online_lora.trainer_torch_dtype == "bf16"
    assert config.online_lora.lora_target_modules == ("q_proj", "v_proj")


def test_environment_config_resolves_lora_base_model_path_from_vllm_server(
    tmp_path,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml(shared_server_model_path="/models/qwen")
        + "online_lora:\n"
        + "  train_batch_size: 1\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.online_lora.base_model == "shared-qwen"
    assert config.online_lora.base_model_path == "/models/qwen"


def test_environment_config_rejects_removed_online_lora_adapter_names(
    tmp_path,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "online_lora:\n"
        + "  interest_adapter_name: interest-lora\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="adapter names are derived"):
        load_environment_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_batch_size", 0),
        ("train_epochs", 0),
        ("max_update_steps", 0),
        ("max_concurrent_trainer_jobs", 0),
        ("max_concurrent_trainer_jobs", -1),
        ("max_update_wait_seconds", 0),
        ("trainer_quantization", "fp8"),
    ],
)
def test_environment_config_rejects_invalid_online_lora_grpo_controls(
    tmp_path,
    field: str,
    value: int | str,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "online_lora:\n"
        + f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"online_lora.{field}"):
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
    dirty_paths = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "--", config_root],
            text=True,
        ).splitlines()
    )
    config_paths = sorted(
        Path(line)
        for line in output.splitlines()
        if line.endswith(".yaml") and line not in dirty_paths
    )

    assert config_paths
    for path in config_paths:
        config = load_environment_config(path)
        assert config.models.agent.backend == "vllm"
        assert config.models.memory.backend == "vllm"
        assert config.models.world.backend == "vllm"
        assert config.models.goal.backend == "vllm"
        assert config.models.reward_judge.backend == "vllm"


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
