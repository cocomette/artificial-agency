"""Smoke tests for model-owned contracts, configs, and adapters."""

from face_of_agi.models import (
    AgentToolRuntime,
    AgentGameContextUpdateInput,
    AgentGameContextUpdaterModel,
    ConfigurableUpdaterAdapter,
    ContextSegment,
    GeneralKnowledgeUpdateInput,
    GeneralKnowledgeUpdaterModel,
    GoalGameContextUpdateInput,
    GoalGameContextUpdaterModel,
    GoalPredictionAdapter,
    OllamaDescriptionConfig,
    GoalPredictionModel,
    ModelRegistry,
    OllamaOrchestratorAgentConfig,
    OllamaUpdaterConfig,
    OpenAIDescriptionConfig,
    OpenAIOrchestratorAgentConfig,
    OpenAIUpdaterConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OrchestratorAgentModel,
    PromptUpdateRequest,
    PromptUpdateResult,
    PromptUpdaterAdapter,
    PromptUpdaterProvider,
    UpdaterContextTarget,
    UpdaterConfig,
    UpdaterRole,
    UpdaterTask,
    UpdaterTaskRegistry,
    VLLMDescriptionConfig,
    VLLMOrchestratorAgentConfig,
    VLLMUpdaterConfig,
    WorldPredictionAdapter,
    WorldGameContextUpdateInput,
    WorldGameContextUpdaterModel,
    WorldPredictionModel,
)
from face_of_agi.models.orchestrator_agent.providers import (
    ConfigurableOrchestratorAgentAdapter,
    HuggingFaceOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent import (
    OllamaOrchestratorAgentAdapter as TopLevelOllamaOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentAdapter as TopLevelOpenAIOrchestratorAgentAdapter,
    VLLMOrchestratorAgentAdapter as TopLevelVLLMOrchestratorAgentAdapter,
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
from face_of_agi.models.goal import GoalPredictionAdapter as DirectGoalPredictionAdapter
from face_of_agi.models.world import WorldPredictionAdapter as DirectWorldPredictionAdapter


def test_model_role_packages_export_contract_config_and_adapter() -> None:
    registry = ModelRegistry()

    assert registry.world_prediction_model is None
    assert OllamaDescriptionConfig().backend == "ollama"
    assert OllamaDescriptionConfig().model == "gemma4:e4b"
    assert OllamaDescriptionConfig().include_output_schema_in_instructions is False
    assert OpenAIDescriptionConfig().backend == "openai"
    assert OpenAIDescriptionConfig().model == "gpt-5-nano"
    assert OpenAIDescriptionConfig().include_output_schema_in_instructions is False
    assert OpenAIDescriptionConfig().reasoning == {"effort": "low"}
    assert OpenAIDescriptionConfig().input_image_size == "1024x1024"
    assert OpenAIDescriptionConfig().input_image_resample == "nearest"
    assert VLLMDescriptionConfig().backend == "vllm"
    assert VLLMDescriptionConfig().model is None
    assert OrchestratorAgentConfig().options == {}
    assert OrchestratorAgentConfig().include_output_schema_in_instructions is False
    assert OpenAIOrchestratorAgentConfig().backend == "openai"
    assert OpenAIOrchestratorAgentConfig().model == "gpt-5-nano"
    assert OpenAIOrchestratorAgentConfig().reasoning == {"effort": "low"}
    assert OllamaOrchestratorAgentConfig().backend == "ollama"
    assert OllamaOrchestratorAgentConfig().model == "gemma4:e4b"
    assert OllamaOrchestratorAgentConfig().think is False
    assert VLLMOrchestratorAgentConfig().backend == "vllm"
    assert VLLMOrchestratorAgentConfig().model is None
    assert UpdaterConfig().backend is None
    assert UpdaterConfig().options == {}
    assert UpdaterConfig().include_output_schema_in_instructions is False
    assert OpenAIUpdaterConfig().model is None
    assert OllamaUpdaterConfig().model is None
    assert VLLMUpdaterConfig().model is None

    # Protocol and adapter names are intentionally imported for package smoke.
    assert WorldPredictionModel is not None
    assert GoalPredictionModel is not None
    assert AgentToolRuntime is not None
    assert AgentGameContextUpdateInput is not None
    assert AgentGameContextUpdaterModel is not None
    assert ContextSegment is not None
    assert GeneralKnowledgeUpdateInput is not None
    assert GeneralKnowledgeUpdaterModel is not None
    assert GoalGameContextUpdateInput is not None
    assert GoalGameContextUpdaterModel is not None
    assert OrchestratorAgentModel is not None
    assert PromptUpdateRequest is not None
    assert PromptUpdateResult is not None
    assert PromptUpdaterAdapter is not None
    assert PromptUpdaterProvider is not None
    assert UpdaterContextTarget is not None
    assert UpdaterRole is not None
    assert UpdaterTask is not None
    assert UpdaterTaskRegistry is not None
    assert WorldGameContextUpdateInput is not None
    assert WorldGameContextUpdaterModel is not None
    assert WorldPredictionAdapter is not None
    assert GoalPredictionAdapter is not None
    assert WorldPredictionAdapter is DirectWorldPredictionAdapter
    assert GoalPredictionAdapter is DirectGoalPredictionAdapter
    assert TopLevelOpenAIOrchestratorAgentAdapter is OpenAIOrchestratorAgentAdapter
    assert TopLevelOllamaOrchestratorAgentAdapter is OllamaOrchestratorAgentAdapter
    assert TopLevelVLLMOrchestratorAgentAdapter is VLLMOrchestratorAgentAdapter
    assert OpenAIOrchestratorAgentAdapter is not None
    assert OllamaOrchestratorAgentAdapter is not None
    assert VLLMOrchestratorAgentAdapter is not None
    assert ConfigurableOrchestratorAgentAdapter is not None
    assert HuggingFaceOrchestratorAgentAdapter is not None
    assert ConfigurableUpdaterAdapter is not None
    assert OrchestratorAgentAdapter is not None
