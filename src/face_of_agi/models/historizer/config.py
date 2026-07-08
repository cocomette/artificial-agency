"""Configuration for the agent context historizer model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.historizer.contracts import (
    agent_context_history_json_schema,
)


def openai_agent_context_history_text_format(
    schema: dict[str, Any] | None = None,
    *,
    name: str = "agent_context_history",
) -> dict[str, Any]:
    """Return the OpenAI Responses text format for historizer outputs."""

    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema or agent_context_history_json_schema(),
        }
    }


def with_openai_agent_context_history_text_format(
    text: dict[str, Any] | None,
    *,
    schema: dict[str, Any] | None = None,
    name: str = "agent_context_history",
) -> dict[str, Any]:
    """Force the OpenAI historizer response schema while preserving text options."""

    return {
        **(text or {}),
        **openai_agent_context_history_text_format(schema, name=name),
    }


@dataclass(slots=True)
class HistorizerConfig:
    """Provider-neutral config for the agent context historizer."""

    backend: str | None = None
    model: str | None = None
    instruction_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    include_output_schema_in_instructions: bool = True


@dataclass(slots=True)
class OpenAIHistorizerConfig(HistorizerConfig):
    """OpenAI-backed historizer config."""

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
    text: dict[str, Any] = field(
        default_factory=openai_agent_context_history_text_format
    )
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
class OllamaHistorizerConfig(HistorizerConfig):
    """Ollama-backed historizer config."""

    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = field(
        default_factory=agent_context_history_json_schema
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
class VLLMHistorizerConfig(HistorizerConfig):
    """vLLM Chat Completions-backed historizer config."""

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
