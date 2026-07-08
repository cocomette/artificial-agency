"""Configuration for transition change summary providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.change.contracts import (
    change_summary_json_schema,
    openai_change_summary_text_format,
)


@dataclass(slots=True)
class OllamaChangeSummaryConfig:
    """Ollama-backed transition change summary config."""

    backend: str | None = "ollama"
    model: str | None = "gemma4:e4b"
    host: str | None = None
    think: bool | str | None = None
    keep_alive: int | str | None = "5m"
    format: str | dict[str, Any] | None = field(
        default_factory=change_summary_json_schema
    )
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0})
    repair_attempts: int = 2
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | None = 4
    animation_frame_budget_coefficient: int = 2
    gaussian_blur_kernel_size: int = 0
    gaussian_noise_deviation: float = 0.0
    activate_diff_mask: bool = False
    activate_bounding_boxes: bool = False
    activate_components: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    dilation_bounding_boxes: int = 3
    width_bounding_boxes: int = 3
    include_output_schema_in_instructions: bool = True


@dataclass(slots=True)
class OpenAIChangeSummaryConfig:
    """OpenAI-backed transition change summary config."""

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
    text: dict[str, Any] = field(default_factory=openai_change_summary_text_format)
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
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | None = 4
    animation_frame_budget_coefficient: int = 2
    gaussian_blur_kernel_size: int = 0
    gaussian_noise_deviation: float = 0.0
    activate_diff_mask: bool = False
    activate_bounding_boxes: bool = False
    activate_components: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    dilation_bounding_boxes: int = 3
    width_bounding_boxes: int = 3
    image_mime_type: str = "image/png"
    include_output_schema_in_instructions: bool = True


@dataclass(slots=True)
class VLLMChangeSummaryConfig:
    """vLLM Chat Completions-backed transition change summary config."""

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
    repair_attempts: int = 2
    input_image_detail: str = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: str = "nearest"
    input_image_crop_arc_grid_edges: int | tuple[int, int, int, int] | None = 4
    animation_frame_budget_coefficient: int = 2
    gaussian_blur_kernel_size: int = 0
    gaussian_noise_deviation: float = 0.0
    activate_diff_mask: bool = False
    activate_bounding_boxes: bool = False
    activate_components: bool = False
    max_nb_components: int = 50
    max_frames_per_call: int = 10
    dilation_bounding_boxes: int = 3
    width_bounding_boxes: int = 3
    image_mime_type: str = "image/png"
    frame_input_mode: str = "image"
    video_fps: float = 1.0
    video_mime_type: str = "video/jpeg"
    include_output_schema_in_instructions: bool = True
