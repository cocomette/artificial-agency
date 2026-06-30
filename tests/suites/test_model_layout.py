"""Smoke tests for the vLLM-only model layout."""

from face_of_agi.models import (
    ChangeSummaryAdapter,
    ModelRegistry,
    ObservationTextConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    PromptUpdaterAdapter,
    PromptUpdaterProvider,
    UpdaterConfig,
    UpdaterTaskRegistry,
    VLLMChangeSummaryConfig,
    VLLMHistorizerAdapter,
    VLLMHistorizerConfig,
    VLLMOrchestratorAgentAdapter,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterAdapter,
    VLLMUpdaterConfig,
)
from face_of_agi.models.change.providers import VLLMChangeSummaryProvider
from face_of_agi.models.historizer.providers import VLLMHistorizerAdapter as ProviderHistorizer
from face_of_agi.models.orchestrator_agent.providers import (
    VLLMOrchestratorAgentAdapter as ProviderAgent,
)
from face_of_agi.models.updater.providers import VLLMUpdaterAdapter as ProviderUpdater


def test_model_role_packages_export_vllm_only_surface() -> None:
    registry = ModelRegistry()

    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is None
    assert registry.agent_context_historizer_model is None
    assert registry.updater_tasks is None
    assert OrchestratorAgentConfig().options == {}
    assert VLLMOrchestratorAgentConfig().backend == "vllm"
    assert VLLMChangeSummaryConfig().backend == "vllm"
    assert VLLMChangeSummaryConfig().max_frames_per_call == 5
    assert VLLMChangeSummaryConfig().reduce_chunk_summaries is True
    assert VLLMChangeSummaryConfig().reducer_keyframe_limit == 6
    assert VLLMHistorizerConfig().backend == "vllm"
    assert VLLMUpdaterConfig().backend == "vllm"
    assert VLLMOrchestratorAgentConfig(
        observation_text={"crop_cells": 2}
    ).observation_text == ObservationTextConfig(crop_cells=2)

    assert ChangeSummaryAdapter is not None
    assert OrchestratorAgentAdapter is not None
    assert PromptUpdaterAdapter is not None
    assert PromptUpdaterProvider is not None
    assert UpdaterConfig is not None
    assert UpdaterTaskRegistry is not None
    assert VLLMChangeSummaryProvider is not None
    assert VLLMHistorizerAdapter is ProviderHistorizer
    assert VLLMOrchestratorAgentAdapter is ProviderAgent
    assert VLLMUpdaterAdapter is ProviderUpdater
