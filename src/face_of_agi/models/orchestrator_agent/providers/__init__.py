"""Provider adapters for orchestrator agent X."""

from face_of_agi.models.orchestrator_agent.providers.configurable import (
    ConfigurableOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.huggingface import (
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

__all__ = [
    "ConfigurableOrchestratorAgentAdapter",
    "HuggingFaceOrchestratorAgentAdapter",
    "OllamaOrchestratorAgentAdapter",
    "OpenAIOrchestratorAgentAdapter",
    "VLLMOrchestratorAgentAdapter",
]
