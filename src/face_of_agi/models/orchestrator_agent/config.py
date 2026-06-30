"""Configuration for orchestrator agent adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.observation_text import ObservationTextConfig


@dataclass(slots=True)
class OrchestratorAgentConfig:
    """Provider-neutral config for the X agent model."""

    backend: str | None = None
    model: str | None = None
    max_tool_calls: int = 0
    repair_attempts: int = 1
    options: dict[str, Any] = field(default_factory=dict)
    include_output_schema_in_instructions: bool = False


@dataclass(slots=True)
class VLLMOrchestratorAgentConfig(OrchestratorAgentConfig):
    """Configuration for the vLLM Chat Completions-backed X agent."""

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
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "2048x2048"
    input_image_resample: str = "nearest"
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    observation_text: ObservationTextConfig | dict[str, Any] = field(
        default_factory=ObservationTextConfig
    )
    extra_request_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.observation_text, dict):
            self.observation_text = ObservationTextConfig(**self.observation_text)
