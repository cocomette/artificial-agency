"""Transition change summary model role."""

from face_of_agi.models.change.adapter import (
    CHANGE_SUMMARY_PROMPT,
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    build_change_summary_prompt,
    change_summary_elements_text,
    change_summary_observation_images,
    load_change_summary_instructions,
    model_visible_any_change_detected,
    model_visible_changed_pixel_count,
    model_visible_changed_pixel_percent,
    parse_change_summary_output,
    validate_change_summary_output,
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
    DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
    DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    change_summary_json_schema,
    openai_change_summary_text_format,
)

__all__ = [
    "ChangeSummaryAdapter",
    "CHANGE_SUMMARY_PROMPT",
    "ChangeSummaryModel",
    "ChangeSummaryOutputError",
    "ChangeSummaryProvider",
    "ChangeSummaryProviderResponse",
    "ChangeSummaryResult",
    "DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS",
    "DEFAULT_CHANGE_SUMMARY_MAX_CHARS",
    "OllamaChangeSummaryConfig",
    "OpenAIChangeSummaryConfig",
    "VLLMChangeSummaryConfig",
    "build_change_summary_prompt",
    "change_summary_elements_text",
    "change_summary_observation_images",
    "change_summary_json_schema",
    "load_change_summary_instructions",
    "model_visible_any_change_detected",
    "model_visible_changed_pixel_count",
    "model_visible_changed_pixel_percent",
    "openai_change_summary_text_format",
    "parse_change_summary_output",
    "validate_change_summary_output",
]
