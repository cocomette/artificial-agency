"""Provider adapters for the agent compacter role."""

from face_of_agi.models.compacter.providers.ollama import (
    OllamaCompacterAdapter,
    OllamaCompacterProvider,
)
from face_of_agi.models.compacter.providers.openai import (
    OpenAICompacterAdapter,
    OpenAICompacterProvider,
)
from face_of_agi.models.compacter.providers.vllm import (
    VLLMCompacterAdapter,
    VLLMCompacterProvider,
)

__all__ = [
    "OllamaCompacterAdapter",
    "OllamaCompacterProvider",
    "OpenAICompacterAdapter",
    "OpenAICompacterProvider",
    "VLLMCompacterAdapter",
    "VLLMCompacterProvider",
]
