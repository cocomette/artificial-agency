"""Provider adapters for updater P."""

from face_of_agi.models.updater.providers.configurable import (
    ConfigurableUpdaterAdapter,
)
from face_of_agi.models.updater.providers.huggingface import (
    HuggingFaceUpdaterAdapter,
)
from face_of_agi.models.updater.providers.ollama import OllamaUpdaterAdapter
from face_of_agi.models.updater.providers.openai import OpenAIUpdaterAdapter
from face_of_agi.models.updater.providers.vllm import VLLMUpdaterAdapter

__all__ = [
    "ConfigurableUpdaterAdapter",
    "HuggingFaceUpdaterAdapter",
    "OllamaUpdaterAdapter",
    "OpenAIUpdaterAdapter",
    "VLLMUpdaterAdapter",
]
