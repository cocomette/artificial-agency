"""Configuration for transition change summary providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.change.contracts import DEFAULT_CHANGE_SUMMARY_MAX_CHARS
from face_of_agi.models.observation_text import ObservationTextConfig
from face_of_agi.models.structured_output import DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS


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
    max_context_tokens: int | None = None
    truncate_context_on_overflow: bool = True
    context_truncation_margin_tokens: int = 256
    context_overflow_retries: int = 3
    options: dict[str, Any] = field(default_factory=dict)
    extra_request_options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "2048x2048"
    input_image_resample: str = "nearest"
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    observation_text: ObservationTextConfig | dict[str, Any] = field(
        default_factory=ObservationTextConfig
    )
    max_frames_per_call: int | None = 5
    reduce_chunk_summaries: bool = True
    reducer_keyframe_limit: int = 6
    include_output_schema_in_instructions: bool = False
    summary_max_chars: int = DEFAULT_CHANGE_SUMMARY_MAX_CHARS
    repair_invalid_output_preview_chars: int = DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS

    def __post_init__(self) -> None:
        if isinstance(self.observation_text, dict):
            self.observation_text = ObservationTextConfig(**self.observation_text)
        self.summary_max_chars = int(self.summary_max_chars)
        if self.summary_max_chars < 1:
            raise ValueError("summary_max_chars must be at least 1")
        self.repair_invalid_output_preview_chars = int(
            self.repair_invalid_output_preview_chars
        )
        if self.max_frames_per_call is not None:
            if isinstance(self.max_frames_per_call, bool):
                raise ValueError("max_frames_per_call must be an integer or null")
            self.max_frames_per_call = int(self.max_frames_per_call)
            if self.max_frames_per_call < 2:
                raise ValueError("max_frames_per_call must be at least 2 or null")
        if not isinstance(self.reduce_chunk_summaries, bool):
            raise ValueError("reduce_chunk_summaries must be a boolean")
        if isinstance(self.reducer_keyframe_limit, bool):
            raise ValueError("reducer_keyframe_limit must be an integer at least 2")
        self.reducer_keyframe_limit = int(self.reducer_keyframe_limit)
        if self.reducer_keyframe_limit < 2:
            raise ValueError("reducer_keyframe_limit must be at least 2")
