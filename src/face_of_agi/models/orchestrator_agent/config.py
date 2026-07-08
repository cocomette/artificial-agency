"""Configuration for orchestrator agent adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class OpenAIOrchestratorAgentConfig(OrchestratorAgentConfig):
    """Configuration for the OpenAI Responses-backed X agent."""

    backend: str | None = "openai"
    model: str | None = "gpt-5-nano"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=lambda: {"effort": "low"})
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    text: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    store: bool | None = None
    service_tier: str | None = None
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = None
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | list[int] | None = None
    frame_scale: int = 4
    extra_request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OllamaOrchestratorAgentConfig(OrchestratorAgentConfig):
    """Configuration for the Ollama-backed local X agent."""

    backend: str | None = "ollama"
    model: str | None = "gemma4:e4b"
    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = None
    keep_alive: str | None = None
    input_image_size: str | tuple[int, int] | None = None
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | list[int] | None = None
    frame_scale: int = 4


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
    use_response_format: bool = False
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = None
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | list[int] | None = None
    image_mime_type: str = "image/png"
    frame_scale: int = 4
    extra_request_options: dict[str, Any] = field(default_factory=dict)
