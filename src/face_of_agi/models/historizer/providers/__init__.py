"""Provider adapters for the historizer model role."""

from face_of_agi.models.historizer.providers.ollama import (
    OllamaHistorizerAdapter,
    OllamaHistorizerProvider,
)
from face_of_agi.models.historizer.providers.openai import (
    OpenAIHistorizerAdapter,
    OpenAIHistorizerProvider,
)
from face_of_agi.models.historizer.providers.vllm import (
    VLLMHistorizerAdapter,
    VLLMHistorizerProvider,
)

__all__ = [
    "OllamaHistorizerAdapter",
    "OllamaHistorizerProvider",
    "OpenAIHistorizerAdapter",
    "OpenAIHistorizerProvider",
    "VLLMHistorizerAdapter",
    "VLLMHistorizerProvider",
]
