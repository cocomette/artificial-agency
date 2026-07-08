"""Configuration for structured description prediction providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.contracts import DESCRIPTION_SCHEMA


def openai_description_response_schema() -> dict[str, Any]:
    """Return the object-root schema required by OpenAI structured outputs."""

    return {
        "type": "object",
        "properties": {
            "items": DESCRIPTION_SCHEMA,
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def openai_description_text_format() -> dict[str, Any]:
    """Return an OpenAI Responses text format for description predictions."""

    return {
        "format": {
            "type": "json_schema",
            "name": "description_prediction",
            "strict": True,
            "schema": openai_description_response_schema(),
        }
    }


@dataclass(slots=True)
class OllamaDescriptionConfig:
    """Ollama-backed structured description prediction config."""

    backend: str | None = "ollama"
    model: str | None = "gemma4:e4b"
    host: str | None = None
    think: bool | str | None = False
    keep_alive: int | str | None = "5m"
    format: str | dict[str, Any] | None = field(default_factory=lambda: DESCRIPTION_SCHEMA)
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})
    repair_attempts: int = 2
    input_image_size: str | tuple[int, int] | None = "256x256"
    input_image_resample: str = "nearest"
    frame_scale: int = 4


@dataclass(slots=True)
class OpenAIDescriptionConfig:
    """OpenAI-backed structured description prediction config."""

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
    text: dict[str, Any] = field(default_factory=openai_description_text_format)
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
    image_mime_type: str = "image/png"
    frame_scale: int = 4
