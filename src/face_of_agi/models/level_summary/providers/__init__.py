"""Provider adapters for the level-summary model role."""

from face_of_agi.models.level_summary.providers.ollama import (
    OllamaLevelSummaryAdapter,
    OllamaLevelSummaryProvider,
)
from face_of_agi.models.level_summary.providers.openai import (
    OpenAILevelSummaryAdapter,
    OpenAILevelSummaryProvider,
)
from face_of_agi.models.level_summary.providers.vllm import (
    VLLMLevelSummaryAdapter,
    VLLMLevelSummaryProvider,
)

__all__ = [
    "OllamaLevelSummaryAdapter",
    "OllamaLevelSummaryProvider",
    "OpenAILevelSummaryAdapter",
    "OpenAILevelSummaryProvider",
    "VLLMLevelSummaryAdapter",
    "VLLMLevelSummaryProvider",
]
