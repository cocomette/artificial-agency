"""Shared structured description prediction capability."""

from face_of_agi.models.description.adapter import DescriptionPredictionAdapter
from face_of_agi.models.description.config import (
    OllamaDescriptionConfig,
    OpenAIDescriptionConfig,
    openai_description_response_schema,
    openai_description_text_format,
)
from face_of_agi.models.description.contracts import (
    DescriptionProvider,
    DescriptionProviderResponse,
    DescriptionRoleSpec,
)

__all__ = [
    "DescriptionPredictionAdapter",
    "DescriptionProvider",
    "DescriptionProviderResponse",
    "DescriptionRoleSpec",
    "OllamaDescriptionConfig",
    "OpenAIDescriptionConfig",
    "openai_description_response_schema",
    "openai_description_text_format",
]
