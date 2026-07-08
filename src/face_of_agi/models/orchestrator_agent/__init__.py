"""Orchestrator agent model package for role X."""

from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderStep,
    AgentToolSpec,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderFunctionCall,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import (
    HFOrchestratorAgentConfig,
    OllamaOrchestratorAgentConfig,
    OpenAIOrchestratorAgentConfig,
    OrchestratorAgentConfig,
    VLLMOrchestratorAgentConfig,
)
from face_of_agi.models.orchestrator_agent.contracts import (
    AgentToolRuntime,
    OrchestratorAgentModel,
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
from face_of_agi.models.orchestrator_agent.providers.hf_transformers import (
    HFOrchestratorAgentAdapter,
)

__all__ = [
    "AgentProviderStep",
    "AgentToolSpec",
    "AgentTurnRequest",
    "AgentToolRuntime",
    "HFOrchestratorAgentAdapter",
    "HFOrchestratorAgentConfig",
    "OllamaOrchestratorAgentAdapter",
    "OllamaOrchestratorAgentConfig",
    "OpenAIOrchestratorAgentAdapter",
    "OpenAIOrchestratorAgentConfig",
    "OrchestratorAgentAdapter",
    "OrchestratorAgentConfig",
    "OrchestratorAgentModel",
    "ProviderFunctionCall",
    "ProviderToolFeedback",
    "VLLMOrchestratorAgentAdapter",
    "VLLMOrchestratorAgentConfig",
]
