"""Configuration for the agent creator model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.agent_creator.contracts import agent_creator_roles_json_schema


@dataclass(slots=True)
class AgentCreatorConfig:
    """Provider-neutral config for the agent creator model."""

    backend: str | None = None
    model: str | None = None
    instruction_path: str | None = None
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "512x512"
    input_image_resample: str = "nearest"
    image_mime_type: str = "image/png"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | None = 4
    options: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 2
    include_output_schema_in_instructions: bool = True


@dataclass(slots=True)
class OllamaAgentCreatorConfig(AgentCreatorConfig):
    """Ollama-backed agent creator config."""

    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = field(
        default_factory=agent_creator_roles_json_schema
    )
    keep_alive: int | str | None = "5m"
    options: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 3000,
        }
    )


@dataclass(slots=True)
class VLLMAgentCreatorConfig(AgentCreatorConfig):
    """vLLM Chat Completions-backed agent creator config."""

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
