"""Runtime/config smoke tests for the active role set."""

from __future__ import annotations

from pathlib import Path

import pytest

from face_of_agi.environment.config import (
    ModelRoleConfig,
    UpdaterRuntimeConfig,
    load_environment_config,
)
from face_of_agi.runtime.shell import _build_model_registry


def test_environment_config_loads_active_model_shape(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(_active_config_yaml(), encoding="utf-8")

    config = load_environment_config(path)

    assert config.models.agent.backend == "vllm"
    assert config.models.change.backend == "vllm"
    assert config.models.compacter.backend == "vllm"
    assert config.models.updater is not None
    assert config.models.updater.agent.backend == "vllm"
    assert config.compacter_action_history_window == 20
    assert config.updater_context_history_window == 20
    assert config.updater_actions_window == 1


def test_environment_config_loads_updater_control_values(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "updater_actions_window: 3\n"
        + "updater_context_history_window: 7\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.updater_actions_window == 3
    assert config.updater_context_history_window == 7


@pytest.mark.parametrize(
    "key",
    [
        "world_action_history_window",
        "compacter_action_history_window",
        "updater_context_history_window",
    ],
)
def test_environment_config_rejects_negative_model_action_history_windows(
    tmp_path,
    key: str,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(_active_config_yaml() + f"{key}: -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_environment_config(path)


@pytest.mark.parametrize("value", [0, -1])
def test_environment_config_rejects_non_positive_action_windows(
    tmp_path,
    value: int,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + f"updater_actions_window: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="updater_actions_window"):
        load_environment_config(path)


def test_environment_config_rejects_removed_model_and_updater_keys(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "  goal:\n"
        + "    backend: vllm\n"
        + "    model: qwen\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.goal"):
        load_environment_config(path)

    path.write_text(
        _active_config_yaml()
        + "  world:\n"
        + "    backend: vllm\n"
        + "    model: qwen\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="models.world"):
        load_environment_config(path)

    path.write_text(
        _active_config_yaml()
        + "  historizer:\n"
        + "    backend: vllm\n"
        + "    model: qwen\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="models.historizer"):
        load_environment_config(path)

    path.write_text(
        _active_config_yaml().replace(
            "    agent:\n      backend: vllm\n      model: qwen\n",
            "    agent_policy:\n      backend: vllm\n      model: qwen\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="models.updater.agent_policy"):
        load_environment_config(path)


def test_committed_runtime_configs_load_without_removed_keys() -> None:
    config_root = Path("src/face_of_agi/runtime/configs")
    config_paths = sorted(config_root.rglob("*.yaml"))

    assert config_paths
    for path in config_paths:
        config = load_environment_config(path)
        assert config.updater_actions_window >= 1
        assert config.compacter_action_history_window >= 0
        assert config.updater_context_history_window >= 0
        assert config.models.updater is not None
        assert config.models.updater.agent.backend is not None


def test_build_model_registry_wires_active_runtime_roles() -> None:
    model = "Qwen/Qwen3.6-35B-A3B-FP8"
    role = ModelRoleConfig(backend="vllm", model=model)
    updater = UpdaterRuntimeConfig(
        agent=ModelRoleConfig(backend="vllm", model=model),
    )

    registry = _build_model_registry(
        agent_config=role,
        change_config=role,
        compacter_config=role,
        shared_vlm_config=ModelRoleConfig(),
        updater_config=updater,
    )

    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is not None
    assert registry.compacter is not None
    assert registry.updater_tasks is not None
    assert registry.updater_tasks.agent_updater is not None


def _active_config_yaml() -> str:
    return (
        "game_index: 0\n"
        "max_actions_per_level: 1\n"
        "operation_mode: offline\n"
        "models:\n"
        "  agent:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  change:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  compacter:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  updater:\n"
        "    agent:\n"
        "      backend: vllm\n"
        "      model: qwen\n"
    )
