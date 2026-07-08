"""Configuration for the agent world-model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.world.contracts import agent_world_model_json_schema


def openai_world_model_text_format(
    schema: dict[str, Any] | None = None,
    *,
    name: str = "agent_world_model",
) -> dict[str, Any]:
    """Return the OpenAI Responses text format for world-model outputs."""

    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema or agent_world_model_json_schema(),
        }
    }


def with_openai_world_model_text_format(
    text: dict[str, Any] | None,
    *,
    schema: dict[str, Any] | None = None,
    name: str = "agent_world_model",
) -> dict[str, Any]:
    """Force the OpenAI world-model response schema while preserving options."""

    return {
        **(text or {}),
        **openai_world_model_text_format(schema, name=name),
    }


@dataclass(slots=True)
class WorldModelConfig:
    """Provider-neutral config for the agent world-model role."""

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


@dataclass(slots=True)
class OpenAIWorldModelConfig(WorldModelConfig):
    """OpenAI-backed world-model config."""

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
    text: dict[str, Any] = field(default_factory=openai_world_model_text_format)
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
class OllamaWorldModelConfig(WorldModelConfig):
    """Ollama-backed world-model config."""

    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = field(
        default_factory=agent_world_model_json_schema
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
class VLLMWorldModelConfig(WorldModelConfig):
    """vLLM Chat Completions-backed world-model config."""

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
    "OllamaWorldModelConfig",
    "OpenAIWorldModelConfig",
    "VLLMWorldModelConfig",
    "WorldModelConfig",
    "openai_world_model_text_format",
    "with_openai_world_model_text_format",
]
