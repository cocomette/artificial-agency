"""Configuration for goal model tool adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.providers.huggingface import ImageEditorPipeline
from face_of_agi.models.providers.openai import OpenAIResponsesImageConfig

GoalImageEditorPipeline = ImageEditorPipeline


@dataclass(slots=True)
class GoalToolConfig:
    """Configuration for local image-editor goal model backends.

    The adapter still exposes the provider-neutral `predict` contract. These
    fields describe Hugging Face Diffusers backends that edit an observation
    image into a goal-relevant predicted or desired observation.
    """

    backend: str | None = "huggingface-diffusers"
    model: str | None = "Qwen/Qwen-Image-Edit"
    pipeline_type: GoalImageEditorPipeline = "qwen_image_edit"
    quantized_model: str | None = None
    quantized_subdir: str = "flux-1-kontext-dev"
    quantize_text_encoder: bool = True
    device: str = "auto"
    torch_dtype: str = "auto"
    seed: int | None = 0
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    max_sequence_length: int = 512
    max_area: int = 1_048_576
    negative_prompt: str = " "
    frame_scale: int = 4
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenAIGoalToolConfig(OpenAIResponsesImageConfig):
    """Configuration for the OpenAI Responses-backed goal model tool."""
