"""Final OpenAI Responses provider-call helpers."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from typing import Any, Literal

from face_of_agi.contracts import Observation
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.image_inputs import observation_to_provider_data_url

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
OpenAIImageDetail = Literal["low", "high", "auto"]
OpenAIInputImageResample = Literal["nearest", "bilinear", "bicubic", "lanczos"]


def json_dumps(value: Any) -> str:
    """Serialize provider payloads deterministically for text model calls."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def object_get(value: Any, key: str, default: Any = None) -> Any:
    """Read a key from SDK objects, dicts, or simple test doubles."""

    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def plain(value: Any) -> Any:
    """Convert SDK models into ordinary dict/list/scalar metadata."""

    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def openai_response_metadata(response: Any | None) -> dict[str, Any]:
    """Return ordinary metadata fields from an OpenAI Responses result."""

    return {
        "response_id": object_get(response, "id"),
        "response_model": object_get(response, "model"),
        "response_status": object_get(response, "status"),
        "usage": plain(object_get(response, "usage")),
        "incomplete_details": plain(object_get(response, "incomplete_details")),
    }


def response_output_text(response: Any) -> str | None:
    """Return SDK-provided output text or reconstruct it from output items."""

    output_text = object_get(response, "output_text")
    if output_text:
        return str(output_text)

    texts: list[str] = []
    for item in object_get(response, "output") or []:
        if object_get(item, "type") != "message":
            continue
        for content in object_get(item, "content") or []:
            if object_get(content, "type") == "output_text":
                text = object_get(content, "text")
                if text:
                    texts.append(str(text))
    if texts:
        return "\n".join(texts)
    return None


def set_optional(target: dict[str, Any], key: str, value: Any) -> None:
    """Set a request field when it carries a meaningful value."""

    if value is None:
        return
    if value == {} or value == []:
        return
    target[key] = value


def apply_response_options(
    request: dict[str, Any],
    config: Any,
    *,
    include_max_tool_calls: bool = True,
    include_parallel_tool_calls: bool = False,
) -> None:
    """Copy common Responses request options from a role config."""

    for key in (
        "reasoning",
        "max_output_tokens",
        "temperature",
        "top_p",
        "text",
        "metadata",
        "store",
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "truncation",
        "include",
    ):
        set_optional(request, key, getattr(config, key, None))
    if include_max_tool_calls:
        set_optional(request, "max_tool_calls", getattr(config, "max_tool_calls", None))
    if include_parallel_tool_calls:
        set_optional(
            request,
            "parallel_tool_calls",
            getattr(config, "parallel_tool_calls", None),
        )
    request.update(getattr(config, "extra_request_options", {}) or {})


class OpenAIResponsesClient:
    """Last-step OpenAI Responses API caller for role-specific adapters."""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client
        self.last_request: dict[str, Any] | None = None

    def create_response(
        self,
        *,
        model: str | None,
        input_items: list[Any],
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        text: dict[str, Any] | None = None,
        include_max_tool_calls: bool = True,
        include_parallel_tool_calls: bool = False,
    ) -> Any:
        """Build and send the final Responses request."""

        request = self.build_request(
            model=model,
            input_items=input_items,
            instructions=instructions,
            tools=tools,
            tool_choice=tool_choice,
            text=text,
            include_max_tool_calls=include_max_tool_calls,
            include_parallel_tool_calls=include_parallel_tool_calls,
        )
        self.last_request = request
        return self._require_client().responses.create(**request)

    def build_request(
        self,
        *,
        model: str | None,
        input_items: list[Any],
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        text: dict[str, Any] | None = None,
        include_max_tool_calls: bool = True,
        include_parallel_tool_calls: bool = False,
    ) -> dict[str, Any]:
        """Build the final Responses request without sending it."""

        if not model:
            raise ValueError("OpenAI Responses calls require an explicit model")
        request: dict[str, Any] = {
            "model": model,
            "input": input_items,
        }
        set_optional(request, "instructions", instructions)
        set_optional(request, "tools", tools)
        set_optional(request, "tool_choice", tool_choice)
        apply_response_options(
            request,
            self.config,
            include_max_tool_calls=include_max_tool_calls,
            include_parallel_tool_calls=include_parallel_tool_calls,
        )
        set_optional(request, "text", text)
        return request

    def _require_client(self) -> Any:
        """Create the OpenAI SDK client lazily."""

        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            set_optional(kwargs, "api_key", self._resolved_api_key())
            set_optional(kwargs, "base_url", getattr(self.config, "base_url", None))
            set_optional(
                kwargs,
                "organization",
                getattr(self.config, "organization", None),
            )
            set_optional(kwargs, "project", getattr(self.config, "project", None))
            set_optional(kwargs, "timeout", getattr(self.config, "timeout", None))
            set_optional(
                kwargs,
                "max_retries",
                getattr(self.config, "max_retries", None),
            )
            set_optional(
                kwargs,
                "default_headers",
                getattr(self.config, "default_headers", None),
            )
            set_optional(
                kwargs,
                "default_query",
                getattr(self.config, "default_query", None),
            )
            self._client = OpenAI(**kwargs)
        return self._client

    def _resolved_api_key(self) -> str | None:
        if getattr(self.config, "api_key", None):
            return self.config.api_key
        api_key_env = getattr(self.config, "api_key_env", None)
        if api_key_env:
            return os.environ.get(api_key_env)
        return None


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
        self._responses = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None

    def generate_image(self, *, prompt: str, observation: Observation) -> OpenAIImageResult:
        """Send one multimodal request and decode the generated image."""

        response = self._responses.create_response(
            model=self.config.model,
            input_items=self._input_items(prompt=prompt, observation=observation),
            instructions=self.config.instructions,
            tools=[self._image_generation_tool()],
            tool_choice={"type": "image_generation"},
            include_parallel_tool_calls=True,
        )
        self.last_request = self._responses.last_request
        return self.parse_response(response)

    def build_request(
        self,
        *,
        prompt: str,
        observation: Observation,
    ) -> dict[str, Any]:
        """Build a Responses API request for one prompt plus observation image."""

        return self._responses.build_request(
            model=self.config.model,
            input_items=self._input_items(prompt=prompt, observation=observation),
            instructions=self.config.instructions,
            tools=[self._image_generation_tool()],
            tool_choice={"type": "image_generation"},
            include_parallel_tool_calls=True,
        )

    def _input_items(
        self,
        *,
        prompt: str,
        observation: Observation,
    ) -> list[dict[str, Any]]:
        """Return final Responses input items for one image-generation call."""

        input_image = self._observation_to_data_url(observation)
        return [
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
        ]

    def parse_response(self, response: Any) -> OpenAIImageResult:
        """Extract the generated image and useful metadata from a response."""

        image_call = self._first_image_generation_call(response)
        if image_call is None:
            response_id = object_get(response, "id")
            raise RuntimeError(
                "OpenAI response did not include an image_generation_call "
                f"result for response {response_id!r}"
            )

        image_base64 = object_get(image_call, "result")
        if not image_base64:
            raise RuntimeError("OpenAI image_generation_call did not include result data")

        image = self._decode_image(str(image_base64))
        output_text = response_output_text(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            "image_model": self.config.image_model,
            "response_id": object_get(response, "id"),
            "response_model": object_get(response, "model"),
            "response_status": object_get(response, "status"),
            "image_generation_call_id": object_get(image_call, "id"),
            "image_generation_status": object_get(image_call, "status"),
            "usage": plain(object_get(response, "usage")),
            "incomplete_details": plain(object_get(response, "incomplete_details")),
            "output_text": output_text,
        }
        return OpenAIImageResult(
            image=image,
            output_text=output_text,
            response_id=object_get(response, "id"),
            metadata=metadata,
        )

    def _image_generation_tool(self) -> dict[str, Any]:
        """Return the configured hosted image-generation tool."""

        tool: dict[str, Any] = {
            "type": "image_generation",
            "model": self.config.image_model,
            "action": self.config.image_action,
        }
        set_optional(tool, "quality", self.config.image_quality)
        set_optional(tool, "size", self.config.image_size)
        set_optional(tool, "output_format", self.config.image_output_format)
        set_optional(
            tool,
            "output_compression",
            self.config.image_output_compression,
        )
        set_optional(tool, "background", self.config.image_background)
        set_optional(tool, "input_fidelity", self.config.image_input_fidelity)
        set_optional(tool, "moderation", self.config.image_moderation)
        set_optional(tool, "partial_images", self.config.image_partial_images)
        tool.update(self.config.image_tool_options)
        return tool

    def _observation_to_data_url(self, observation: Observation) -> str:
        """Encode the observation frame as a base64 data URL."""

        return observation_to_provider_data_url(
            observation,
            frame_scale=self.config.frame_scale,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
            mime_type=self.config.image_mime_type,
        )

    def _observation_to_image(self, observation: Observation) -> Any:
        """Normalize a framework observation frame into a PIL RGB image."""

        return observation_to_pil_image(
            observation,
            frame_scale=self.config.frame_scale,
        )

    def _first_image_generation_call(self, response: Any) -> Any | None:
        """Return the first image generation output item."""

        for item in self._get(response, "output") or []:
            if self._get(item, "type") == "image_generation_call":
                return item
        return None

    def _decode_image(self, image_base64: str) -> Any:
        """Decode raw base64 or data URL image data to a PIL RGB image."""

        from PIL import Image

        if image_base64.startswith("data:"):
            _, image_base64 = image_base64.split(",", 1)
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _get(self, value: Any, key: str) -> Any:
        """Read a key from SDK objects, dicts, or simple test doubles."""

        return object_get(value, key)
