"""Agent compacter role."""

from face_of_agi.models.compacter.adapter import (
    AgentCompacterAdapter,
    CompacterOutputError,
    load_compacter_instructions,
    parse_agent_compacter_output,
)
from face_of_agi.models.compacter.config import (
    OllamaCompacterConfig,
    OpenAICompacterConfig,
    VLLMCompacterConfig,
    CompacterConfig,
    openai_compacter_text_format,
)
from face_of_agi.models.compacter.contracts import (
    AgentCompacterModel,
    AgentCompacterSummary,
    AgentCompacterInput,
    PromptCompacterImage,
    PromptCompacterProvider,
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
    agent_compacter_json_schema,
)

__all__ = [
    "AgentCompacterSummary",
    "AgentCompacterModel",
    "AgentCompacterAdapter",
    "AgentCompacterInput",
    "OllamaCompacterConfig",
    "OpenAICompacterConfig",
    "PromptCompacterImage",
    "PromptCompacterProvider",
    "PromptCompacterProviderResponse",
    "PromptCompacterRequest",
    "VLLMCompacterConfig",
    "CompacterOutputError",
    "CompacterConfig",
    "agent_compacter_json_schema",
    "load_compacter_instructions",
    "openai_compacter_text_format",
    "parse_agent_compacter_output",
]
