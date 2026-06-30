"""Smoke tests for vLLM-only runtime wiring."""

from io import StringIO
from pathlib import Path

import pytest

from face_of_agi.environment.config import (
    EnvironmentConfig,
    ModelRuntimeConfig,
    ModelRoleConfig,
    UpdaterRuntimeConfig,
    load_environment_config,
)
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.updater.providers.vllm import VLLMUpdaterAdapter
import face_of_agi.runtime.kaggle as kaggle
from face_of_agi.runtime import shell
from face_of_agi.runtime.parallel import ParallelGameRunSpec


def _vllm_role(**options: object) -> ModelRoleConfig:
    return ModelRoleConfig(
        backend="vllm",
        model="fake-vllm",
        options=dict(options),
    )


def _updater_config() -> UpdaterRuntimeConfig:
    return UpdaterRuntimeConfig(agent=_vllm_role(), general=_vllm_role())


def test_shell_model_registry_wires_vllm_roles_and_observation_text() -> None:
    registry = shell._build_model_registry(
        agent_config=_vllm_role(),
        change_config=_vllm_role(),
        historizer_config=_vllm_role(),
        updater_config=_updater_config(),
        observation_text_config={
            "crop_cells": 2,
            "overflow_chars_per_frame": 99,
            "include_rows": False,
            "include_components": False,
            "include_component_runs": False,
            "compact_components": True,
        },
    )

    assert isinstance(registry.orchestrator_agent, VLLMOrchestratorAgentAdapter)
    assert registry.change_summary_model is not None
    assert registry.agent_context_historizer_model is not None
    assert registry.updater_tasks is not None
    assert isinstance(registry.updater_tasks.agent_game_updater, VLLMUpdaterAdapter)
    assert registry.orchestrator_agent.config.observation_text.crop_cells == 2
    assert (
        registry.orchestrator_agent.config.observation_text.include_components
        is False
    )
    assert registry.orchestrator_agent.config.observation_text.include_rows is False
    assert (
        registry.orchestrator_agent.config.observation_text.include_component_runs
        is False
    )
    assert registry.orchestrator_agent.config.observation_text.compact_components is True
    assert registry.change_summary_model.config.observation_text.crop_cells == 2
    assert registry.change_summary_model.config.observation_text.include_rows is False
    assert (
        registry.change_summary_model.config.observation_text.compact_components
        is True
    )
    assert registry.updater_tasks.agent_game_updater.config.observation_text == (
        registry.orchestrator_agent.config.observation_text
    )


def test_shared_vllm_server_max_model_len_becomes_role_context_limit() -> None:
    shared = ModelRoleConfig(
        backend="vllm",
        model="fake-vllm",
        options={
            "server": {"max_model_len": 12345},
            "temperature": 0.0,
        },
    )
    role = ModelRoleConfig(backend="vllm")

    merged = shell._with_shared_vlm_role_config(role, shared)

    assert merged.options["max_context_tokens"] == 12345
    assert merged.options["temperature"] == 0.0
    assert "server" not in merged.options


def test_shell_rejects_removed_change_frame_budget_key() -> None:
    with pytest.raises(ValueError, match="max_evidence_frames.*max_frames_per_call"):
        shell._build_model_registry(
            agent_config=_vllm_role(),
            change_config=_vllm_role(max_evidence_frames=5),
            historizer_config=_vllm_role(),
            updater_config=_updater_config(),
        )


@pytest.mark.parametrize("backend", ["openai", "ollama", "huggingface", "diffusers"])
def test_shell_rejects_removed_real_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="use vllm"):
        shell._build_model_registry(
            agent_config=ModelRoleConfig(backend=backend, model="old"),
            change_config=_vllm_role(),
            historizer_config=_vllm_role(),
            updater_config=_updater_config(),
        )


def test_environment_config_loads_shared_observation_text(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
game_index: 0
max_actions_per_level: 1
models:
  observation_text:
    crop_cells: 2
    overflow_chars_per_frame: 77
    include_rows: false
    include_components: false
    include_component_runs: false
    compact_components: true
  shared_vlm:
    backend: vllm
    model: fake-vllm
  agent:
    backend: vllm
  change:
    backend: vllm
  historizer:
    backend: vllm
  updater:
    agent:
      backend: vllm
    general:
      backend: vllm
""".lstrip(),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.models.observation_text == {
        "crop_cells": 2,
        "overflow_chars_per_frame": 77,
        "include_rows": False,
        "include_components": False,
        "include_component_runs": False,
        "compact_components": True,
    }
    assert config.models.agent.backend == "vllm"


def test_kaggle_worker_wires_shared_observation_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observation_text = {
        "crop_cells": 3,
        "overflow_chars_per_frame": 123,
        "include_rows": True,
        "include_components": True,
        "include_component_runs": True,
        "compact_components": True,
    }
    captured: dict[str, object] = {}

    class StopAfterRegistry(Exception):
        pass

    def fake_build_model_registry(**kwargs: object) -> object:
        captured.update(kwargs)
        raise StopAfterRegistry

    monkeypatch.setattr(kaggle, "_build_model_registry", fake_build_model_registry)
    environment_config = EnvironmentConfig(
        game_id="fake-game",
        max_actions_per_level=1,
        models=ModelRuntimeConfig(
            observation_text=observation_text,
            shared_vlm=ModelRoleConfig(backend="vllm", model="fake-vllm"),
            agent=ModelRoleConfig(backend="vllm"),
            change=ModelRoleConfig(backend="vllm"),
            historizer=ModelRoleConfig(backend="vllm"),
            updater=UpdaterRuntimeConfig(
                agent=ModelRoleConfig(backend="vllm"),
                general=ModelRoleConfig(backend="vllm"),
            ),
        ),
    )
    spec = ParallelGameRunSpec(
        game_index=0,
        game_id="fake-game",
        run_id="run-1",
        database_path=tmp_path / "memory.sqlite",
        environment_config=environment_config,
        arc_environment=object(),
    )

    with pytest.raises(StopAfterRegistry):
        kaggle._run_kaggle_game(spec, StringIO())

    assert captured["observation_text_config"] == observation_text


def test_rtx6000_configs_load_multimodal_runtime_contract() -> None:
    config_dir = Path("src/face_of_agi/runtime/configs/vllm")
    debug_config = load_environment_config(
        config_dir / "vllm_rtx6000_qwen36_35b_fp8_debug.yaml"
    )
    parallel_config = load_environment_config(
        config_dir / "vllm_rtx6000_qwen36_35b_fp8_parallel.yaml"
    )

    assert debug_config.models.observation_text["crop_cells"] == 3
    assert parallel_config.models.observation_text["crop_cells"] == 3
    assert debug_config.models.observation_text["overflow_chars_per_frame"] > 0
    assert parallel_config.models.observation_text["overflow_chars_per_frame"] > 0
    assert debug_config.models.observation_text["include_rows"] is True
    assert debug_config.models.observation_text["include_components"] is True
    assert debug_config.models.observation_text["include_component_runs"] is True
    assert debug_config.models.observation_text["compact_components"] is True
    assert parallel_config.models.observation_text["include_rows"] is True
    assert parallel_config.models.observation_text["include_components"] is True
    assert parallel_config.models.observation_text["include_component_runs"] is True
    assert parallel_config.models.observation_text["compact_components"] is True
    assert debug_config.models.shared_vlm.options["server"]["max_model_len"] >= 65536
    assert parallel_config.models.shared_vlm.options["server"]["max_model_len"] == 65536
    assert debug_config.models.shared_vlm.options["input_image_size"] == "1024x1024"
    assert parallel_config.models.shared_vlm.options["input_image_size"] == "1024x1024"
    assert debug_config.models.change.options["max_frames_per_call"] >= 2
    assert parallel_config.models.change.options["max_frames_per_call"] >= 2
    assert debug_config.models.change.options["reduce_chunk_summaries"] is True
    assert parallel_config.models.change.options["reduce_chunk_summaries"] is True
    assert debug_config.models.change.options["reducer_keyframe_limit"] == 8
    assert parallel_config.models.change.options["reducer_keyframe_limit"] == 8
