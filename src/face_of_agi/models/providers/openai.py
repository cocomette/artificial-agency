"""Shared OpenAI Responses image backend for model tools."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import os
from typing import Any, Literal

from face_of_agi.contracts import Observation
from face_of_agi.frames import image_to_data_url, observation_to_pil_image

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
OpenAIImageDetail = Literal["low", "high", "auto"]
OpenAIInputImageResample = Literal["nearest", "bilinear", "bicubic", "lanczos"]


@dataclass(slots=True)
class OpenAIResponsesImageConfig:
    """Configuration for Responses API calls that return generated images."""

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

    model: str = "gpt-5-nano"
    instructions: str | None = None
    reasoning: dict[str, Any] = field(
        default_factory=lambda: {"effort": "low"}
    )
    max_output_tokens: int | None = None
    max_tool_calls: int | None = 1
    temperature: float | None = None
    top_p: float | None = None
    text: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    store: bool | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    include: list[str] = field(default_factory=list)
    extra_request_options: dict[str, Any] = field(default_factory=dict)

    input_image_detail: OpenAIImageDetail = "auto"
    input_image_size: str | tuple[int, int] | None = "1024x1024"
    input_image_resample: OpenAIInputImageResample = "nearest"
    image_mime_type: str = "image/png"
    frame_scale: int = 4

    image_model: str = "gpt-image-1-mini"
    image_action: str = "edit"
    image_quality: str | None = "low"
    image_size: str | None = "1024x1024"
    image_output_format: str | None = "png"
    image_output_compression: int | None = None
    image_background: str | None = None
    image_input_fidelity: str | None = None
    image_moderation: str | None = None
    image_partial_images: int | None = None
    image_tool_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenAIImageResult:
    """Normalized image-generation response from OpenAI."""

    image: Any
    output_text: str | None
    response_id: str | None
    metadata: dict[str, Any]


class OpenAIImageGenerationClient:
    """Thin wrapper around OpenAI Responses image generation calls."""

    def __init__(
        self,
        config: OpenAIResponsesImageConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self.last_request: dict[str, Any] | None = None

    def generate_image(self, *, prompt: str, observation: Observation) -> OpenAIImageResult:
        """Send one multimodal request and decode the generated image."""

        request = self.build_request(prompt=prompt, observation=observation)
        self.last_request = request
        response = self._require_client().responses.create(**request)
        return self.parse_response(response)

    def build_request(
        self,
        *,
        prompt: str,
        observation: Observation,
    ) -> dict[str, Any]:
        """Build a Responses API request for one prompt plus observation image."""

        input_image = self._observation_to_data_url(observation)
        request: dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": input_image,
                            "detail": self.config.input_image_detail,
                        },
                    ],
                }
            ],
            "tools": [self._image_generation_tool()],
            "tool_choice": {"type": "image_generation"},
        }

        self._set_optional(request, "instructions", self.config.instructions)
        self._set_optional(request, "reasoning", self.config.reasoning)
        self._set_optional(request, "max_output_tokens", self.config.max_output_tokens)
        self._set_optional(request, "max_tool_calls", self.config.max_tool_calls)
        self._set_optional(request, "temperature", self.config.temperature)
        self._set_optional(request, "top_p", self.config.top_p)
        self._set_optional(request, "text", self.config.text)
        self._set_optional(request, "metadata", self.config.metadata)
        self._set_optional(request, "store", self.config.store)
        self._set_optional(request, "service_tier", self.config.service_tier)
        self._set_optional(request, "prompt_cache_key", self.config.prompt_cache_key)
        self._set_optional(
            request,
            "prompt_cache_retention",
            self.config.prompt_cache_retention,
        )
        self._set_optional(request, "safety_identifier", self.config.safety_identifier)
        self._set_optional(request, "truncation", self.config.truncation)
        self._set_optional(
            request,
            "parallel_tool_calls",
            self.config.parallel_tool_calls,
        )
        self._set_optional(request, "include", self.config.include)
        request.update(self.config.extra_request_options)
        return request

    def parse_response(self, response: Any) -> OpenAIImageResult:
        """Extract the generated image and useful metadata from a response."""

        image_call = self._first_image_generation_call(response)
        if image_call is None:
            response_id = self._get(response, "id")
            raise RuntimeError(
                "OpenAI response did not include an image_generation_call "
                f"result for response {response_id!r}"
            )

        image_base64 = self._get(image_call, "result")
        if not image_base64:
            raise RuntimeError("OpenAI image_generation_call did not include result data")

        image = self._decode_image(str(image_base64))
        output_text = self._response_output_text(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            "image_model": self.config.image_model,
            "response_id": self._get(response, "id"),
            "response_model": self._get(response, "model"),
            "response_status": self._get(response, "status"),
            "image_generation_call_id": self._get(image_call, "id"),
            "image_generation_status": self._get(image_call, "status"),
            "usage": self._plain(self._get(response, "usage")),
            "incomplete_details": self._plain(self._get(response, "incomplete_details")),
            "output_text": output_text,
        }
        return OpenAIImageResult(
            image=image,
            output_text=output_text,
            response_id=self._get(response, "id"),
            metadata=metadata,
        )

    def _require_client(self) -> Any:
        """Create the OpenAI SDK client lazily."""

        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            api_key = self._resolved_api_key()
            self._set_optional(kwargs, "api_key", api_key)
            self._set_optional(kwargs, "base_url", self.config.base_url)
            self._set_optional(kwargs, "organization", self.config.organization)
            self._set_optional(kwargs, "project", self.config.project)
            self._set_optional(kwargs, "timeout", self.config.timeout)
            self._set_optional(kwargs, "max_retries", self.config.max_retries)
            self._set_optional(
                kwargs,
                "default_headers",
                self.config.default_headers,
            )
            self._set_optional(kwargs, "default_query", self.config.default_query)
            self._client = OpenAI(**kwargs)
        return self._client

    def _resolved_api_key(self) -> str | None:
        """Return the explicit API key or configured environment value."""

        if self.config.api_key:
            return self.config.api_key
        if self.config.api_key_env:
            return os.environ.get(self.config.api_key_env)
        return None

    def _image_generation_tool(self) -> dict[str, Any]:
        """Return the configured hosted image-generation tool."""

        tool: dict[str, Any] = {
            "type": "image_generation",
            "model": self.config.image_model,
            "action": self.config.image_action,
        }
        self._set_optional(tool, "quality", self.config.image_quality)
        self._set_optional(tool, "size", self.config.image_size)
        self._set_optional(tool, "output_format", self.config.image_output_format)
        self._set_optional(
            tool,
            "output_compression",
            self.config.image_output_compression,
        )
        self._set_optional(tool, "background", self.config.image_background)
        self._set_optional(tool, "input_fidelity", self.config.image_input_fidelity)
        self._set_optional(tool, "moderation", self.config.image_moderation)
        self._set_optional(tool, "partial_images", self.config.image_partial_images)
        tool.update(self.config.image_tool_options)
        return tool

    def _observation_to_data_url(self, observation: Observation) -> str:
        """Encode the observation frame as a base64 data URL."""

        image = self._resize_input_image_if_needed(
            observation_to_pil_image(
                observation,
                frame_scale=self.config.frame_scale,
            )
        )
        if self.config.image_mime_type == "image/png":
            return image_to_data_url(image)

        buffer = BytesIO()
        image.save(buffer, format=self._pil_format())
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{self.config.image_mime_type};base64,{encoded}"

    def _observation_to_image(self, observation: Observation) -> Any:
        """Normalize a framework observation frame into a PIL RGB image."""

        return observation_to_pil_image(
            observation,
            frame_scale=self.config.frame_scale,
        )

    def _resize_input_image_if_needed(self, image: Any) -> Any:
        """Resize the encoded input image when a provider config requests it."""

        target_size = self._input_image_size()
        if target_size is None or image.size == target_size:
            return image
        return image.resize(target_size, self._input_image_resampling())

    def _input_image_size(self) -> tuple[int, int] | None:
        """Return the configured input image size as a PIL size tuple."""

        size = self.config.input_image_size
        if size is None:
            return None
        if isinstance(size, tuple) and len(size) == 2:
            width, height = size
        elif isinstance(size, str) and "x" in size:
            width_text, height_text = size.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        else:
            raise ValueError(
                "input_image_size must be None, a (width, height) tuple, "
                "or a string like '1024x1024'"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"input_image_size must be positive, got {size!r}")
        return (width, height)

    def _input_image_resampling(self) -> Any:
        """Return the PIL resampling filter for configured input resizing."""

        from PIL import Image

        filters = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }
        return filters[self.config.input_image_resample]

    def _pil_format(self) -> str:
        """Return the PIL save format matching the configured MIME type."""

        if self.config.image_mime_type == "image/jpeg":
            return "JPEG"
        if self.config.image_mime_type == "image/webp":
            return "WEBP"
        return "PNG"

    def _first_image_generation_call(self, response: Any) -> Any | None:
        """Return the first image generation output item."""

        for item in self._get(response, "output") or []:
            if self._get(item, "type") == "image_generation_call":
                return item
        return None

    def _response_output_text(self, response: Any) -> str | None:
        """Return SDK-provided output text or reconstruct it from output items."""

        output_text = self._get(response, "output_text")
        if output_text:
            return str(output_text)

        texts: list[str] = []
        for item in self._get(response, "output") or []:
            if self._get(item, "type") != "message":
                continue
            for content in self._get(item, "content") or []:
                if self._get(content, "type") == "output_text":
                    text = self._get(content, "text")
                    if text:
                        texts.append(str(text))
        if texts:
            return "\n".join(texts)
        return None

    def _decode_image(self, image_base64: str) -> Any:
        """Decode raw base64 or data URL image data to a PIL RGB image."""

        from PIL import Image

        if image_base64.startswith("data:"):
            _, image_base64 = image_base64.split(",", 1)
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _plain(self, value: Any) -> Any:
        """Convert SDK models into ordinary dict/list/scalar metadata."""

        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return {key: self._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._plain(item) for item in value]
        return value

    def _get(self, value: Any, key: str) -> Any:
        """Read a key from SDK objects, dicts, or simple test doubles."""

        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _set_optional(self, target: dict[str, Any], key: str, value: Any) -> None:
        """Set a request field when it carries a meaningful value."""

        if value is None:
            return
        if value == {} or value == []:
            return
        target[key] = value
