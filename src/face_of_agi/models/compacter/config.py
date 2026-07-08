"""Configuration for the agent compacter role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.compacter.contracts import agent_compacter_json_schema


def openai_compacter_text_format(
    schema: dict[str, Any] | None = None,
    *,
    name: str = "agent_compacter",
) -> dict[str, Any]:
    """Return the OpenAI Responses text format for compacter outputs."""

    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema or agent_compacter_json_schema(),
        }
    }


def with_openai_compacter_text_format(
    text: dict[str, Any] | None,
    *,
    schema: dict[str, Any] | None = None,
    name: str = "agent_compacter",
) -> dict[str, Any]:
    """Force the OpenAI compacter response schema while preserving options."""

    return {
        **(text or {}),
        **openai_compacter_text_format(schema, name=name),
    }


@dataclass(slots=True)
class CompacterConfig:
    """Provider-neutral config for the agent compacter role."""

    backend: str | None = None
    model: str | None = None
    instruction_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    include_output_schema_in_instructions: bool = True
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | None = 4
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = None
    input_image_resample: str = "nearest"
    image_mime_type: str = "image/png"
    max_nb_components: int = 50


@dataclass(slots=True)
class OpenAICompacterConfig(CompacterConfig):
    """OpenAI-backed compacter config."""

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
    text: dict[str, Any] = field(default_factory=openai_compacter_text_format)
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
class OllamaCompacterConfig(CompacterConfig):
    """Ollama-backed compacter config."""

    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = field(
        default_factory=agent_compacter_json_schema
    )
    keep_alive: int | str | None = "5m"
    options: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 1000,
        }
    )


@dataclass(slots=True)
class VLLMCompacterConfig(CompacterConfig):
    """vLLM Chat Completions-backed compacter config."""

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


__all__ = [
    "OllamaCompacterConfig",
    "OpenAICompacterConfig",
    "VLLMCompacterConfig",
    "CompacterConfig",
    "openai_compacter_text_format",
    "with_openai_compacter_text_format",
]
