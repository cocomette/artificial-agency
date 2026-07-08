"""Configuration for the game memory model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.memory.contracts import (
    GAME_MEMORY_MAX_CHARS,
    game_memory_json_schema,
    openai_game_memory_text_format,
)
from face_of_agi.models.structured_output import DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS


def with_openai_game_memory_text_format(
    text: dict[str, Any] | None,
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Force the OpenAI game-memory response schema while preserving options."""

    return {
        **(text or {}),
        **openai_game_memory_text_format(schema=schema),
    }


@dataclass(slots=True)
class GameMemoryConfig:
    """Provider-neutral config for game memory generation."""

    backend: str | None = None
    model: str | None = None
    instruction_path: str | None = None
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_box_normalized: tuple[float, float, float, float] | None = None
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 1
    include_output_schema_in_instructions: bool = False
    memory_max_chars: int | None = GAME_MEMORY_MAX_CHARS
    repair_invalid_output_preview_chars: int | None = (
        DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS
    )

    def __post_init__(self) -> None:
        """Validate optional output-size controls."""

        if self.memory_max_chars is not None and self.memory_max_chars <= 0:
            raise ValueError("memory_max_chars must be positive or None")
        if (
            self.repair_invalid_output_preview_chars is not None
            and self.repair_invalid_output_preview_chars < 0
        ):
            raise ValueError(
                "repair_invalid_output_preview_chars must be non-negative or None"
            )


@dataclass(slots=True)
class OpenAIGameMemoryConfig(GameMemoryConfig):
    """OpenAI-backed game memory config."""

    backend: str | None = "openai"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    text: dict[str, Any] = field(default_factory=openai_game_memory_text_format)
    metadata: dict[str, Any] = field(default_factory=dict)
    store: bool | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    truncation: str | None = None
    include: list[str] = field(default_factory=list)
    extra_request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OllamaGameMemoryConfig(GameMemoryConfig):
    """Ollama-backed game memory config."""

    backend: str | None = "ollama"
    model: str | None = None
    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = field(
        default_factory=game_memory_json_schema
    )
    keep_alive: int | str | None = "5m"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})


@dataclass(slots=True)
class VLLMGameMemoryConfig(GameMemoryConfig):
    """vLLM Chat Completions-backed game memory config."""

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
    scheduler: Any | None = None
    scheduler_queue_timeout_seconds: float | None = None
