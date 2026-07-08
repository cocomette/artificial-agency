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
    assert config.models.world.backend == "vllm"
    assert config.models.historizer.backend == "vllm"
    assert config.models.updater is not None
    assert config.models.updater.agent_probing.backend == "vllm"
    assert config.models.updater.agent_policy.backend == "vllm"
    assert config.models.updater.general.backend == "vllm"
    assert config.probing_actions_window == 1
    assert config.policy_actions_window == 1
    assert config.probing_mode_cap_ratio == 0.35
    assert config.world_action_history_window == 8
    assert config.historizer_action_history_window == 8
    assert config.probing_action_history_window == 8
    assert config.policy_action_history_window == 8


def test_environment_config_loads_updater_control_values(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "probing_actions_window: 3\n"
        + "policy_actions_window: 2\n"
        + "probing_mode_cap_ratio: 0.4\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.probing_actions_window == 3
    assert config.policy_actions_window == 2
    assert config.probing_mode_cap_ratio == 0.4


def test_environment_config_loads_model_action_history_windows(tmp_path) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml()
        + "world_action_history_window: 20\n"
        + "historizer_action_history_window: 20\n"
        + "probing_action_history_window: 5\n"
        + "policy_action_history_window: 3\n",
        encoding="utf-8",
    )

    config = load_environment_config(path)

    assert config.world_action_history_window == 20
    assert config.historizer_action_history_window == 20
    assert config.probing_action_history_window == 5
    assert config.policy_action_history_window == 3


@pytest.mark.parametrize(
    "key",
    [
        "world_action_history_window",
        "historizer_action_history_window",
        "probing_action_history_window",
        "policy_action_history_window",
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("probing_actions_window", 0),
        ("probing_actions_window", -1),
        ("policy_actions_window", 0),
        ("policy_actions_window", -1),
    ],
)
def test_environment_config_rejects_non_positive_action_windows(
    tmp_path,
    key: str,
    value: int,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(_active_config_yaml() + f"{key}: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_environment_config(path)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_environment_config_rejects_invalid_probing_mode_cap_ratio(
    tmp_path,
    value: float,
) -> None:
    path = tmp_path / "active.yaml"
    path.write_text(
        _active_config_yaml() + f"probing_mode_cap_ratio: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probing_mode_cap_ratio"):
        load_environment_config(path)


def test_committed_runtime_configs_load_without_removed_keys() -> None:
    config_root = Path("src/face_of_agi/runtime/configs")
    config_paths = sorted(config_root.rglob("*.yaml"))

    assert config_paths
    for path in config_paths:
        config = load_environment_config(path)
        assert config.probing_actions_window >= 1
        assert config.policy_actions_window >= 1
        assert config.world_action_history_window >= 0
        assert config.historizer_action_history_window >= 0
        assert config.probing_action_history_window >= 0
        assert config.policy_action_history_window >= 0
        assert 0 <= config.probing_mode_cap_ratio <= 1
        assert config.models.updater is not None
        assert config.models.updater.agent_probing.backend is not None
        assert config.models.updater.agent_policy.backend is not None
        assert config.models.updater.general.backend is not None


def test_build_model_registry_wires_active_runtime_roles() -> None:
    model = "Qwen/Qwen3.6-35B-A3B-FP8"
    role = ModelRoleConfig(backend="vllm", model=model)
    updater = UpdaterRuntimeConfig(
        agent_probing=ModelRoleConfig(backend="vllm", model=model),
        agent_policy=ModelRoleConfig(backend="vllm", model=model),
        general=ModelRoleConfig(backend="vllm", model=model),
    )

    registry = _build_model_registry(
        agent_config=role,
        change_config=role,
        world_config=role,
        historizer_config=role,
        level_summary_config=role,
        shared_vlm_config=ModelRoleConfig(),
        updater_config=updater,
    )

    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is not None
    assert registry.world_model is not None
    assert registry.agent_context_historizer_model is not None
    assert registry.level_solution_summarizer is not None
    assert registry.updater_tasks is not None
    assert registry.updater_tasks.agent_probing_updater is not None
    assert registry.updater_tasks.agent_policy_updater is not None
    assert registry.updater_tasks.general_updater is not None


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
        "  world:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  historizer:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  level_summary:\n"
        "    backend: vllm\n"
        "    model: qwen\n"
        "  updater:\n"
        "    agent_probing:\n"
        "      backend: vllm\n"
        "      model: qwen\n"
        "    agent_policy:\n"
        "      backend: vllm\n"
        "      model: qwen\n"
        "    general:\n"
        "      backend: vllm\n"
        "      model: qwen\n"
    )
