"""Configuration for the agent context historizer model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.historizer.contracts import DEFAULT_FIELD_EVOLUTION_MAX_CHARS
from face_of_agi.models.structured_output import DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS


@dataclass(slots=True)
class HistorizerConfig:
    """Provider-neutral config for the agent context historizer."""

    backend: str | None = None
    model: str | None = None
    instruction_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    include_output_schema_in_instructions: bool = False
    field_evolution_max_chars: int = DEFAULT_FIELD_EVOLUTION_MAX_CHARS
    repair_invalid_output_preview_chars: int = DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS

    def __post_init__(self) -> None:
        self.field_evolution_max_chars = int(self.field_evolution_max_chars)
        if self.field_evolution_max_chars < 1:
            raise ValueError("field_evolution_max_chars must be at least 1")
        self.repair_invalid_output_preview_chars = int(
            self.repair_invalid_output_preview_chars
        )


@dataclass(slots=True)
class VLLMHistorizerConfig(HistorizerConfig):
    """vLLM Chat Completions-backed historizer config."""

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
