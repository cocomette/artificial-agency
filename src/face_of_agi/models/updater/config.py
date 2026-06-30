"""Configuration for updater model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.observation_text import ObservationTextConfig
from face_of_agi.models.structured_output import DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_FIELD_MAX_CHARS,
    AGENT_GAME_CONTEXT_MAX_CHARS,
    GENERAL_CONTEXT_MAX_CHARS,
)


@dataclass(slots=True)
class UpdaterConfig:
    """Provider-neutral config for the updater model."""

    backend: str | None = None
    model: str | None = None
    instruction_dir: str | None = None
    observation_text: ObservationTextConfig | dict[str, Any] = field(
        default_factory=ObservationTextConfig
    )
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "2048x2048"
    input_image_resample: str = "nearest"
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    include_output_schema_in_instructions: bool = False
    general_context_max_chars: int = GENERAL_CONTEXT_MAX_CHARS
    agent_game_context_max_chars: int = AGENT_GAME_CONTEXT_MAX_CHARS
    agent_game_context_field_max_chars: int = AGENT_GAME_CONTEXT_FIELD_MAX_CHARS
    repair_invalid_output_preview_chars: int = DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS

    def __post_init__(self) -> None:
        if isinstance(self.observation_text, dict):
            self.observation_text = ObservationTextConfig(**self.observation_text)
        self.general_context_max_chars = int(self.general_context_max_chars)
        if self.general_context_max_chars < 1:
            raise ValueError("general_context_max_chars must be at least 1")
        self.agent_game_context_max_chars = int(self.agent_game_context_max_chars)
        if self.agent_game_context_max_chars < 1:
            raise ValueError("agent_game_context_max_chars must be at least 1")
        self.agent_game_context_field_max_chars = int(
            self.agent_game_context_field_max_chars
        )
        if self.agent_game_context_field_max_chars < 1:
            raise ValueError("agent_game_context_field_max_chars must be at least 1")
        self.repair_invalid_output_preview_chars = int(
            self.repair_invalid_output_preview_chars
        )


@dataclass(slots=True)
class VLLMUpdaterConfig(UpdaterConfig):
    """vLLM Chat Completions-backed updater config.

    The model is intentionally not defaulted here. Runtime config must name it.
    """

    backend: str | None = "vllm"
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
