"""Smoke tests for current model-owned contracts, configs, and adapters."""

from face_of_agi.models import (
    AGENT_CONTEXT_HISTORY_KEYS,
    AgentContextHistorizerAdapter,
    AgentGameContextUpdateInput,
    AgentGameContextUpdaterModel,
    AgentToolRuntime,
    ChangeSummaryAdapter,
    ChangeSummaryResult,
    ContextSegment,
    GameMemoryAdapter,
    GameMemoryDocument,
    GameMemoryInput,
    GeneralKnowledgeUpdateInput,
    GeneralKnowledgeUpdaterModel,
    ModelRegistry,
    OllamaChangeSummaryConfig,
    OllamaGameMemoryConfig,
    OllamaHistorizerConfig,
    OllamaOrchestratorAgentConfig,
    OllamaUpdaterConfig,
    OpenAIChangeSummaryConfig,
    OpenAIGameMemoryConfig,
    OpenAIHistorizerConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OrchestratorAgentModel,
    PromptGameMemoryRequest,
    PromptHistorizerRequest,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    PromptUpdaterProvider,
    UpdaterContextTarget,
    UpdaterConfig,
    UpdaterRole,
    UpdaterTask,
    UpdaterTaskRegistry,
    VLLMChangeSummaryConfig,
    VLLMGameMemoryConfig,
    VLLMHistorizerConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
)


def test_current_model_role_packages_export_contract_config_and_adapter() -> None:
    registry = ModelRegistry()

    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is None
    assert registry.agent_context_historizer_model is None
    assert registry.game_memory_model is None
    assert registry.updater_tasks is None

    assert OrchestratorAgentConfig().options == {}
    assert OpenAIOrchestratorAgentConfig().backend == "openai"
    assert OllamaOrchestratorAgentConfig().backend == "ollama"
    assert VLLMOrchestratorAgentConfig().backend == "vllm"
    assert VLLMOrchestratorAgentConfig().scheduler is None

    assert OllamaChangeSummaryConfig().summary_max_chars == 2000
    assert OpenAIChangeSummaryConfig().summary_max_chars == 2000
    assert VLLMChangeSummaryConfig().summary_max_elements == 20
    assert VLLMChangeSummaryConfig().repair_invalid_output_preview_chars == 8000

    assert OllamaHistorizerConfig().field_max_chars == 2000
    assert OpenAIHistorizerConfig().field_max_chars == 2000
    assert VLLMHistorizerConfig().repair_invalid_output_preview_chars == 8000

    assert OllamaGameMemoryConfig().memory_max_chars == 10000
    assert OpenAIGameMemoryConfig().memory_max_chars == 10000
    assert VLLMGameMemoryConfig().repair_invalid_output_preview_chars == 8000

    assert UpdaterConfig().backend is None
    assert OpenAIUpdaterConfig().general_context_max_chars == 20000
    assert OllamaUpdaterConfig().agent_game_context_field_max_chars == 6000
    assert VLLMUpdaterConfig().repair_invalid_output_preview_chars == 8000

    assert AgentContextHistorizerAdapter is not None
    assert AgentGameContextUpdateInput is not None
    assert AgentGameContextUpdaterModel is not None
    assert AgentToolRuntime is not None
    assert AGENT_CONTEXT_HISTORY_KEYS
    assert ChangeSummaryAdapter is not None
    assert ChangeSummaryResult is not None
    assert ContextSegment is not None
    assert GameMemoryAdapter is not None
    assert GameMemoryDocument.not_available().is_available() is False
    assert GameMemoryInput is not None
    assert GeneralKnowledgeUpdateInput is not None
    assert GeneralKnowledgeUpdaterModel is not None
    assert OrchestratorAgentAdapter is not None
    assert OrchestratorAgentModel is not None
    assert PromptGameMemoryRequest is not None
    assert PromptHistorizerRequest is not None
    assert PromptUpdateRequest is not None
    assert PromptUpdaterAdapter is not None
    assert PromptUpdaterProvider is not None
    assert UpdaterContextTarget is not None
    assert UpdaterRole is not None
    assert UpdaterTask is not None
    assert UpdaterTaskRegistry is not None
