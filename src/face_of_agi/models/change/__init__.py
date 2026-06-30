"""Transition change summary model role."""

from face_of_agi.models.change.adapter import (
    CHANGE_SUMMARY_PROMPT,
    CHANGE_SUMMARY_REDUCER_PROMPT,
    ChangeSummaryAdapter,
    ChangeSummaryOutputError,
    build_change_summary_prompt,
    build_change_summary_reducer_prompt,
    cropped_changed_cell_percent,
    load_change_summary_instructions,
    load_change_summary_reducer_instructions,
    parse_change_summary_output,
    validate_change_summary_output,
)
from face_of_agi.models.change.config import VLLMChangeSummaryConfig
from face_of_agi.models.change.contracts import (
    ChangeSummaryModel,
    ChangeSummaryProvider,
    ChangeSummaryProviderResponse,
    ChangeSummaryResult,
    DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    change_summary_json_schema,
)

__all__ = [
    "ChangeSummaryAdapter",
    "CHANGE_SUMMARY_PROMPT",
    "CHANGE_SUMMARY_REDUCER_PROMPT",
    "ChangeSummaryModel",
    "ChangeSummaryOutputError",
    "ChangeSummaryProvider",
    "ChangeSummaryProviderResponse",
    "ChangeSummaryResult",
    "DEFAULT_CHANGE_SUMMARY_MAX_CHARS",
    "VLLMChangeSummaryConfig",
    "build_change_summary_prompt",
    "build_change_summary_reducer_prompt",
    "change_summary_json_schema",
    "cropped_changed_cell_percent",
    "load_change_summary_instructions",
    "load_change_summary_reducer_instructions",
    "parse_change_summary_output",
    "validate_change_summary_output",
]
