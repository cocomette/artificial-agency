"""Smoke tests for model-owned contracts, configs, and adapters."""

from face_of_agi.models import (
    AgentToolRuntime,
    AgentContextUpdateInput,
    GoalImageEditorPipeline,
    GoalToolAdapter,
    GoalToolConfig,
    GoalToolModel,
    ModelRegistry,
    OllamaOrchestratorAgentConfig,
    OpenAIGoalToolConfig,
    OpenAIGoalToolAdapter as TopLevelOpenAIGoalToolAdapter,
    OpenAIOrchestratorAgentConfig,
    OpenAIWorldToolAdapter as TopLevelOpenAIWorldToolAdapter,
    OpenAIWorldToolConfig,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OrchestratorAgentModel,
    ToolContextUpdateInput,
    UpdaterAdapter,
    UpdaterConfig,
    UpdaterModel,
    WorldToolAdapter,
    WorldToolConfig,
    WorldToolModel,
)
from face_of_agi.models.orchestrator_agent.providers import (
    ConfigurableOrchestratorAgentAdapter,
    HuggingFaceOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent import (
    OllamaOrchestratorAgentAdapter as TopLevelOllamaOrchestratorAgentAdapter,
    OpenAIOrchestratorAgentAdapter as TopLevelOpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.random import (
    RandomOrchestratorAgentAdapter,
)
from face_of_agi.models.tools.goal.providers import (
    ConfigurableGoalToolAdapter,
    HuggingFaceGoalToolAdapter,
)
from face_of_agi.models.tools.goal.providers.openai import OpenAIGoalToolAdapter
from face_of_agi.models.tools.world.providers import (
    ConfigurableWorldToolAdapter,
    HuggingFaceWorldToolAdapter,
)
from face_of_agi.models.tools.world.providers.openai import OpenAIWorldToolAdapter


def test_model_role_packages_export_contract_config_and_adapter() -> None:
    registry = ModelRegistry()

    assert registry.world_tool is None
    assert WorldToolConfig().backend == "huggingface-diffusers"
    assert WorldToolConfig().model == "Qwen/Qwen-Image-Edit"
    assert GoalToolConfig().backend == "huggingface-diffusers"
    assert GoalToolConfig().model == "Qwen/Qwen-Image-Edit"
    assert OpenAIWorldToolConfig().backend == "openai"
    assert OpenAIWorldToolConfig().model == "gpt-5-nano"
    assert OpenAIWorldToolConfig().reasoning == {"effort": "low"}
    assert OpenAIWorldToolConfig().input_image_size == "1024x1024"
    assert OpenAIWorldToolConfig().input_image_resample == "nearest"
    assert OpenAIWorldToolConfig().image_model == "gpt-image-1-mini"
    assert OpenAIWorldToolConfig().image_size == "1024x1024"
    assert OpenAIWorldToolConfig().image_quality == "low"
    assert OpenAIGoalToolConfig().backend == "openai"
    assert OpenAIGoalToolConfig().model == "gpt-5-nano"
    assert OpenAIGoalToolConfig().reasoning == {"effort": "low"}
    assert OpenAIGoalToolConfig().input_image_size == "1024x1024"
    assert OpenAIGoalToolConfig().input_image_resample == "nearest"
    assert OpenAIGoalToolConfig().image_model == "gpt-image-1-mini"
    assert OpenAIGoalToolConfig().image_size == "1024x1024"
    assert OpenAIGoalToolConfig().image_quality == "low"
    assert OrchestratorAgentConfig().options == {}
    assert OpenAIOrchestratorAgentConfig().backend == "openai"
    assert OpenAIOrchestratorAgentConfig().model == "gpt-5-nano"
    assert OpenAIOrchestratorAgentConfig().reasoning == {"effort": "low"}
    assert OllamaOrchestratorAgentConfig().backend == "ollama"
    assert OllamaOrchestratorAgentConfig().model == "gemma4:e4b"
    assert OllamaOrchestratorAgentConfig().think is False
    assert UpdaterConfig().options == {}

    # Protocol and adapter names are intentionally imported for package smoke.
    assert GoalImageEditorPipeline is not None
    assert WorldToolModel is not None
    assert GoalToolModel is not None
    assert AgentToolRuntime is not None
    assert AgentContextUpdateInput is not None
    assert OrchestratorAgentModel is not None
    assert ToolContextUpdateInput is not None
    assert UpdaterModel is not None
    assert WorldToolAdapter is not None
    assert GoalToolAdapter is not None
    assert TopLevelOpenAIWorldToolAdapter is OpenAIWorldToolAdapter
    assert TopLevelOpenAIGoalToolAdapter is OpenAIGoalToolAdapter
    assert OpenAIWorldToolAdapter is not None
    assert OpenAIGoalToolAdapter is not None
    assert TopLevelOpenAIOrchestratorAgentAdapter is OpenAIOrchestratorAgentAdapter
    assert TopLevelOllamaOrchestratorAgentAdapter is OllamaOrchestratorAgentAdapter
    assert OpenAIOrchestratorAgentAdapter is not None
    assert OllamaOrchestratorAgentAdapter is not None
    assert ConfigurableGoalToolAdapter is not None
    assert ConfigurableOrchestratorAgentAdapter is not None
    assert ConfigurableWorldToolAdapter is not None
    assert HuggingFaceGoalToolAdapter is not None
    assert HuggingFaceOrchestratorAgentAdapter is not None
    assert HuggingFaceWorldToolAdapter is not None
    assert RandomOrchestratorAgentAdapter is not None
    assert OrchestratorAgentAdapter is not None
    assert UpdaterAdapter is not None
