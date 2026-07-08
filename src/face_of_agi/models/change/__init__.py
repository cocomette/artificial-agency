"""Transition change summary model role."""

from face_of_agi.models.change.adapter import (
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    build_change_summary_prompt,
    change_summary_elements_text,
    load_change_summary_instructions,
    parse_change_summary_output,
)
from face_of_agi.models.change.config import (
    OllamaChangeSummaryConfig,
    OpenAIChangeSummaryConfig,
    VLLMChangeSummaryConfig,
)
from face_of_agi.models.change.contracts import (
    ChangeSummaryModel,
    ChangeSummaryProvider,
    ChangeSummaryProviderResponse,
    ChangeSummaryResult,
    change_summary_json_schema,
    openai_change_summary_text_format,
)

__all__ = [
    "ChangeSummaryAdapter",
    "ChangeSummaryModel",
    "ChangeSummaryOutputError",
    "ChangeSummaryProvider",
    "ChangeSummaryProviderResponse",
    "ChangeSummaryResult",
    "OllamaChangeSummaryConfig",
    "OpenAIChangeSummaryConfig",
    "VLLMChangeSummaryConfig",
    "build_change_summary_prompt",
    "change_summary_elements_text",
    "change_summary_json_schema",
    "load_change_summary_instructions",
    "openai_change_summary_text_format",
    "parse_change_summary_output",
]
