"""World prediction model S package."""

from face_of_agi.models.description import (
    OllamaDescriptionConfig,
    OpenAIDescriptionConfig,
    VLLMDescriptionConfig,
)
from face_of_agi.models.world.adapter import (
    WORLD_DESCRIPTION_ROLE,
    WorldPredictionAdapter,
)
from face_of_agi.models.world.contracts import WorldPredictionModel

__all__ = [
    "OllamaDescriptionConfig",
    "OpenAIDescriptionConfig",
    "VLLMDescriptionConfig",
    "WORLD_DESCRIPTION_ROLE",
    "WorldPredictionAdapter",
    "WorldPredictionModel",
]
