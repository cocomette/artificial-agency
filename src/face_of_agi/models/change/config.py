"""Configuration for transition change summary providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.change.contracts import (
    DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    change_summary_json_schema,
    openai_change_summary_text_format,
)
from face_of_agi.models.structured_output import DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS


@dataclass(slots=True)
class OllamaChangeSummaryConfig:
    """Ollama-backed transition change summary config."""

    backend: str | None = "ollama"
    model: str | None = "gemma4:e4b"
    host: str | None = None
    think: bool | str | None = None
    keep_alive: int | str | None = "5m"
    format: str | dict[str, Any] | None = field(
        default_factory=change_summary_json_schema
    )
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})
    repair_attempts: int = 2
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_box_normalized: tuple[float, float, float, float] | None = None
    frame_scale: int = 4
    include_output_schema_in_instructions: bool = False
    activate_components: bool = False
    persist_changed_elements_only: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS
    summary_max_elements: int | None = 20


@dataclass(slots=True)
class OpenAIChangeSummaryConfig:
    """OpenAI-backed transition change summary config."""

    backend: str | None = "openai"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)
    model: str | None = "gpt-5-nano"
    instructions: str | None = None
    reasoning: dict[str, Any] = field(default_factory=lambda: {"effort": "low"})
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    text: dict[str, Any] = field(default_factory=openai_change_summary_text_format)
    metadata: dict[str, Any] = field(default_factory=dict)
    store: bool | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    truncation: str | None = None
    include: list[str] = field(default_factory=list)
    extra_request_options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_box_normalized: tuple[float, float, float, float] | None = None
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    include_output_schema_in_instructions: bool = False
    activate_components: bool = False
    persist_changed_elements_only: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS
    summary_max_elements: int | None = 20


@dataclass(slots=True)
class VLLMChangeSummaryConfig:
    """vLLM Chat Completions-backed transition change summary config."""

    backend: str | None = "vllm"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = "VLLM_API_KEY"
    base_url: str = "http://127.0.0.1:8000/v1"
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    extra_request_options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_box_normalized: tuple[float, float, float, float] | None = None
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    include_output_schema_in_instructions: bool = False
    activate_components: bool = False
    persist_changed_elements_only: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS
    summary_max_elements: int | None = 20
    repair_invalid_output_preview_chars: int | None = (
        DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS
    )
    scheduler: Any | None = None
    scheduler_queue_timeout_seconds: float | None = None
