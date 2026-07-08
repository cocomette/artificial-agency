"""Smoke tests for active model-owned contracts, configs, and adapters."""

import json

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
)
from face_of_agi.models import (
    AgentCompacterAdapter,
    AgentCompacterInput,
    AgentCompacterModel,
    AgentCompacterSummary,
    AgentContextUpdaterModel,
    AgentGameContextUpdateInput,
    AgentToolRuntime,
    ChangeSummaryAdapter,
    ChangeSummaryModel,
    ConfigurableUpdaterAdapter,
    ContextSegment,
    ModelRegistry,
    OllamaChangeSummaryConfig,
    OllamaCompacterConfig,
    OllamaOrchestratorAgentConfig,
    OllamaUpdaterConfig,
    OpenAIChangeSummaryConfig,
    OpenAICompacterAdapter,
    OpenAICompacterConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OrchestratorAgentModel,
    PromptUpdateRequest,
    PromptUpdateResult,
    PromptUpdaterAdapter,
    PromptUpdaterProvider,
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
    UpdaterConfig,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    UpdaterTaskRegistry,
    VLLMChangeSummaryConfig,
    VLLMCompacterConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
    CompacterConfig,
)
from face_of_agi.models.orchestrator_agent import (
    OllamaOrchestratorAgentAdapter as TopLevelOllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent import (
    OpenAIOrchestratorAgentAdapter as TopLevelOpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent import (
    VLLMOrchestratorAgentAdapter as TopLevelVLLMOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers import (
    ConfigurableOrchestratorAgentAdapter,
    HuggingFaceOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)


def test_model_role_packages_export_active_contract_config_and_adapter() -> None:
    registry = ModelRegistry()

    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is None
    assert registry.updater_tasks is None

    assert CompacterConfig().backend is None
    assert OllamaCompacterConfig().model is None
    assert OpenAICompacterConfig().model is None
    assert VLLMCompacterConfig().model is None
    assert OllamaChangeSummaryConfig().backend == "ollama"
    assert OpenAIChangeSummaryConfig().backend == "openai"
    assert VLLMChangeSummaryConfig().backend == "vllm"
    assert OrchestratorAgentConfig().options == {}
    assert OpenAIOrchestratorAgentConfig().backend == "openai"
    assert OllamaOrchestratorAgentConfig().backend == "ollama"
    assert VLLMOrchestratorAgentConfig().backend == "vllm"
    assert UpdaterConfig().backend is None
    assert UpdaterConfig().max_nb_components == 50
    assert OpenAIUpdaterConfig().model is None
    assert OllamaUpdaterConfig().model is None
    assert VLLMUpdaterConfig().model is None
    assert CompacterConfig().max_nb_components == 50

    assert ChangeSummaryAdapter is not None
    assert ChangeSummaryModel is not None
    assert AgentToolRuntime is not None
    assert AgentCompacterAdapter is not None
    assert AgentCompacterModel is not None
    assert AgentCompacterSummary is not None
    assert AgentCompacterInput is not None
    assert OpenAICompacterAdapter is not None
    assert AgentContextUpdaterModel is not None
    assert AgentGameContextUpdateInput is not None
    assert ContextSegment is not None
    assert OrchestratorAgentModel is not None
    assert PromptUpdateRequest is not None
    assert PromptUpdateResult is not None
    assert PromptUpdaterAdapter is not None
    assert PromptUpdaterProvider is not None
    assert UpdaterContextTarget is not None
    assert UpdaterRole is not None
    assert UpdaterTask is not None
    assert UpdaterTaskRegistry is not None
    assert TopLevelOpenAIOrchestratorAgentAdapter is OpenAIOrchestratorAgentAdapter
    assert TopLevelOllamaOrchestratorAgentAdapter is OllamaOrchestratorAgentAdapter
    assert TopLevelVLLMOrchestratorAgentAdapter is VLLMOrchestratorAgentAdapter
    assert ConfigurableOrchestratorAgentAdapter is not None
    assert HuggingFaceOrchestratorAgentAdapter is not None
    assert ConfigurableUpdaterAdapter is not None
    assert OrchestratorAgentAdapter is not None


class _FakeCompacterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[PromptCompacterRequest] = []

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        self.requests.append(request)
        return PromptCompacterProviderResponse(
            text=json.dumps(
                {
                    "world_description": "world",
                    "special_events": "",
                    "action_effects": {"ACTION1": "moves"},
                    "previous_actions_summary": "actions",
                    "previous_strategy_summary": "strategies",
                }
            )
        )

    def repair_compacter_context(
        self, *args, **kwargs
    ) -> PromptCompacterProviderResponse:
        raise AssertionError("unexpected repair")


def test_compacter_prompt_includes_windowed_context(
    tmp_path,
) -> None:
    instruction_path = tmp_path / "compacter_prompt.md"
    instruction_path.write_text("Compact context.", encoding="utf-8")
    provider = _FakeCompacterProvider()
    adapter = AgentCompacterAdapter(
        provider,
        CompacterConfig(instruction_path=instruction_path),
    )

    adapter.compact_agent_context(
        AgentCompacterInput(
            game_id="game-1",
            current_observation=Observation(id="obs-1", step=1, frame=[[0]]),
            action_history=(
                ActionHistoryEntry(
                    action=ActionSpec(action_id="ACTION1"),
                    controllable=True,
                    changed_pixel_count=1.0,
                    change_summary="first step",
                ),
                ActionHistoryEntry(
                    action=ActionSpec(action_id="ACTION2"),
                    controllable=True,
                    changed_pixel_count=1.0,
                    change_summary="latest step",
                ),
            ),
            strategy_history=("first plan", "latest plan"),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            previous_compacter_context=(
                '{\n'
                '  "world_description": "existing world facts",\n'
                '  "special_events": "",\n'
                '  "action_effects": {"ACTION1": "prior effect"},\n'
                '  "previous_actions_summary": "prior compact actions",\n'
                '  "previous_strategy_summary": "prior compact strategy"\n'
                "}"
            ),
        )
    )

    text = provider.requests[0].text
    assert "## Previous compacter context" in text
    assert "existing world facts" in text
    assert "prior compact actions" in text
    assert "first step" in text
    assert "latest step" in text
    assert "1.\n   first plan" in text
    assert "2. [latest]\n   latest plan" in text
