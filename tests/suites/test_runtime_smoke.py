"""Smoke tests for current runtime/config boundaries."""

from pathlib import Path

import pytest

from face_of_agi.environment.config import ModelRoleConfig, load_environment_config
from face_of_agi.models import DisabledGameMemoryAdapter
from face_of_agi.runtime import shell


def test_rtx6000_parallel_config_preserves_run063_retry_and_memory_values() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_parallel.yaml"
    )

    assert config.max_parallel_games == 25
    assert config.max_game_retries == 0
    assert config.models.memory.backend == "vllm"
    assert config.models.memory.options["memory_max_chars"] == 10000
    assert config.models.change.options["summary_max_chars"] == 2000
    assert config.models.updater.agent.options["agent_game_context_max_chars"] == 12000


def test_rtx6000_debug_config_sets_repair_attempts_and_caps() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_debug.yaml"
    )

    assert config.models.shared_vlm.repair_attempts == 3
    assert config.models.memory.options["repair_invalid_output_preview_chars"] == 8000
    assert config.models.historizer.options["field_max_chars"] == 2000
    assert config.models.updater.general.options["general_context_max_chars"] == 20000


def test_removed_world_goal_model_config_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
max_actions_per_level: 10
game_index: 0
models:
  world:
    backend: openai
  shared_vlm:
    backend: vllm
  agent:
    backend: random
  change:
    backend: vllm
    model: model
  memory:
    backend: vllm
    model: model
  updater:
    agent:
      backend: vllm
      model: model
    general:
      backend: vllm
      model: model
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.world config has been removed"):
        load_environment_config(config_path)


def test_memory_backend_none_builds_disabled_adapter() -> None:
    model = shell._build_game_memory_model(ModelRoleConfig(backend="none"))

    assert isinstance(model, DisabledGameMemoryAdapter)


def test_shared_vlm_caps_expand_into_role_dataclasses() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_debug.yaml"
    )
    registry = shell._build_model_registry(
        agent_config=config.models.agent,
        change_config=config.models.change,
        historizer_config=config.models.historizer,
        memory_config=config.models.memory,
        shared_vlm_config=config.models.shared_vlm,
        scheduler_config=config.models.scheduler,
        updater_config=config.models.updater,
    )

    assert registry.change_summary_model.config.summary_max_chars == 2000
    assert registry.change_summary_model.config.summary_max_elements == 20
    assert registry.agent_context_historizer_model.config.field_max_chars == 2000
    assert registry.game_memory_model.config.memory_max_chars == 10000
    assert (
        registry.updater_tasks.agent_game_updater.config.agent_game_context_max_chars
        == 12000
    )


def test_tuned_rtx6000_config_enables_scheduler_and_computed_timeouts() -> None:
    config = load_environment_config(
        "src/face_of_agi/runtime/configs/vllm/"
        "vllm_rtx6000_qwen36_35b_fp8_tuned.yaml"
    )

    assert config.max_parallel_games == 25
    assert config.models.scheduler.enabled is True
    assert config.models.scheduler.max_concurrent_calls == 8
    assert config.models.scheduler.max_concurrent_calls_per_game == 1
    assert config.models.change.options["summary_max_elements"] == 20

    registry = shell._build_model_registry(
        agent_config=config.models.agent,
        change_config=config.models.change,
        historizer_config=config.models.historizer,
        memory_config=config.models.memory,
        shared_vlm_config=config.models.shared_vlm,
        scheduler_config=config.models.scheduler,
        updater_config=config.models.updater,
    )

    assert registry.orchestrator_agent.config.timeout == 92.0
    assert registry.change_summary_model.config.timeout == 124.0
    assert registry.game_memory_model.config.timeout == 124.0
    assert registry.agent_context_historizer_model.config.timeout == 90.0
    assert registry.updater_tasks.agent_game_updater.config.timeout == 188.0
    assert registry.updater_tasks.general_updater.config.timeout == 90.0
    assert registry.orchestrator_agent.config.scheduler is not None
    assert "scheduler" not in registry.orchestrator_agent.config.options
    assert (
        "scheduler_queue_timeout_seconds"
        not in registry.orchestrator_agent.config.options
    )
    assert registry.change_summary_model.config.scheduler is not None
