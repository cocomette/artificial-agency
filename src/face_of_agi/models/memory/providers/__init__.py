"""Provider adapters for the game memory role."""

from face_of_agi.models.memory.providers.ollama import OllamaGameMemoryAdapter
from face_of_agi.models.memory.providers.openai import OpenAIGameMemoryAdapter
from face_of_agi.models.memory.providers.vllm import VLLMGameMemoryAdapter

__all__ = [
    "OllamaGameMemoryAdapter",
    "OpenAIGameMemoryAdapter",
    "VLLMGameMemoryAdapter",
]
