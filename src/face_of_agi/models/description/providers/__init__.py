"""Concrete providers for structured description predictions."""

from face_of_agi.models.description.providers.ollama import OllamaDescriptionProvider
from face_of_agi.models.description.providers.openai import OpenAIDescriptionProvider
from face_of_agi.models.description.providers.vllm import VLLMDescriptionProvider

__all__ = [
    "OllamaDescriptionProvider",
    "OpenAIDescriptionProvider",
    "VLLMDescriptionProvider",
]
