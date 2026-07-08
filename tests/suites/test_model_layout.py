"""Smoke tests for active model-owned contracts, configs, and adapters."""

from face_of_agi.models import (
    AgentContextHistorizerAdapter,
    AgentContextHistorizerModel,
    AgentContextHistoryInput,
    AgentContextHistorySummary,
    LevelSolutionSummarizerAdapter,
    LevelSolutionSummarizerModel,
    LevelSolutionSummary,
    LevelSolutionSummaryInput,
    LevelSummaryConfig,
    OllamaLevelSummaryConfig,
    OpenAILevelSummaryAdapter,
    OpenAILevelSummaryConfig,
    VLLMLevelSummaryConfig,
    AgentProbingContextUpdaterModel,
    AgentPolicyContextUpdaterModel,
    AgentGameContextUpdateInput,
    AgentToolRuntime,
    ChangeSummaryAdapter,
    ChangeSummaryModel,
    ConfigurableUpdaterAdapter,
    ContextSegment,
    GeneralKnowledgeUpdateInput,
    GeneralKnowledgeUpdaterModel,
    ModelRegistry,
    OllamaChangeSummaryConfig,
    OllamaHistorizerConfig,
    OllamaOrchestratorAgentConfig,
    OllamaUpdaterConfig,
    OpenAIChangeSummaryConfig,
    OpenAIHistorizerConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OrchestratorAgentModel,
    PromptUpdateRequest,
    PromptUpdateResult,
    PromptUpdaterAdapter,
    PromptUpdaterProvider,
    UpdaterConfig,
    UpdaterContextTarget,
    UpdaterRole,
    UpdaterTask,
    UpdaterTaskRegistry,
    VLLMChangeSummaryConfig,
    VLLMHistorizerConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
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

    assert registry.agent_context_historizer_model is None
    assert registry.orchestrator_agent is None
    assert registry.change_summary_model is None
    assert registry.updater_tasks is None

    assert OpenAIHistorizerConfig().backend is None
    assert OllamaHistorizerConfig().backend is None
    assert VLLMHistorizerConfig().backend is None
    assert LevelSummaryConfig().backend is None
    assert OllamaLevelSummaryConfig().model is None
    assert OpenAILevelSummaryConfig().model is None
    assert VLLMLevelSummaryConfig().model is None
    assert OllamaChangeSummaryConfig().backend == "ollama"
    assert OpenAIChangeSummaryConfig().backend == "openai"
    assert VLLMChangeSummaryConfig().backend == "vllm"
    assert OrchestratorAgentConfig().options == {}
    assert OpenAIOrchestratorAgentConfig().backend == "openai"
    assert OllamaOrchestratorAgentConfig().backend == "ollama"
    assert VLLMOrchestratorAgentConfig().backend == "vllm"
    assert UpdaterConfig().backend is None
    assert OpenAIUpdaterConfig().model is None
    assert OllamaUpdaterConfig().model is None
    assert VLLMUpdaterConfig().model is None

    assert ChangeSummaryAdapter is not None
    assert ChangeSummaryModel is not None
    assert AgentToolRuntime is not None
    assert AgentContextHistorizerAdapter is not None
    assert AgentContextHistorizerModel is not None
    assert AgentContextHistoryInput is not None
    assert AgentContextHistorySummary is not None
    assert LevelSolutionSummarizerAdapter is not None
    assert LevelSolutionSummarizerModel is not None
    assert LevelSolutionSummary is not None
    assert LevelSolutionSummaryInput is not None
    assert OpenAILevelSummaryAdapter is not None
    assert AgentProbingContextUpdaterModel is not None
    assert AgentPolicyContextUpdaterModel is not None
    assert AgentGameContextUpdateInput is not None
    assert ContextSegment is not None
    assert GeneralKnowledgeUpdateInput is not None
    assert GeneralKnowledgeUpdaterModel is not None
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
