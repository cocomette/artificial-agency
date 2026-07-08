"""Final vLLM OpenAI-compatible chat provider-call helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from face_of_agi.models.providers.openai import object_get, plain, set_optional

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_API_KEY_ENV = "VLLM_API_KEY"


@dataclass(slots=True)
class VLLMChatConfig:
    """Minimal shared config for vLLM Chat Completions calls."""

    backend: str | None = "vllm"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = DEFAULT_VLLM_API_KEY_ENV
    base_url: str = DEFAULT_VLLM_BASE_URL
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


class VLLMChatClient:
    """Last-step vLLM Chat Completions caller for role-specific adapters."""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client
        self.last_request: dict[str, Any] | None = None

    def chat(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """Build and send the final vLLM chat request."""

        request = self.build_request(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_body=extra_body,
        )
        self.last_request = request
        return self._require_client().chat.completions.create(**request)

    def build_request(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the final vLLM chat request without sending it."""

        if not model:
            raise ValueError("vLLM chat calls require an explicit model")
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        set_optional(request, "tools", tools)
        set_optional(request, "tool_choice", tool_choice)
        set_optional(request, "response_format", response_format)
        for key in (
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "seed",
        ):
            set_optional(request, key, getattr(self.config, key, None))
        merged_extra_body = {
            **(getattr(self.config, "options", None) or {}),
            **(extra_body or {}),
        }
        set_optional(request, "extra_body", merged_extra_body)
        request.update(getattr(self.config, "extra_request_options", {}) or {})
        return request

    def _require_client(self) -> Any:
        """Create the OpenAI SDK client lazily for vLLM."""

        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            set_optional(kwargs, "api_key", self._resolved_api_key())
            set_optional(kwargs, "base_url", getattr(self.config, "base_url", None))
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

    def _resolved_api_key(self) -> str:
        if getattr(self.config, "api_key", None):
            return str(self.config.api_key)
        api_key_env = getattr(self.config, "api_key_env", None)
        if api_key_env:
            value = os.environ.get(str(api_key_env))
            if value:
                return value
        return "EMPTY"


def json_schema_response_format(
    *,
    name: str,
    schema: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    """Return a Chat Completions JSON-schema response_format."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def chat_message(response: Any) -> Any:
    """Return the first chat completion message from an SDK/dict response."""

    choices = object_get(response, "choices") or []
    if not choices:
        raise RuntimeError("vLLM chat response did not include choices")
    return object_get(choices[0], "message") or {}


def chat_message_content(response: Any) -> str:
    """Extract required assistant message content from a chat response."""

    content = chat_message_optional_content(response)
    if content is None:
        raise RuntimeError("vLLM chat response did not include message content")
    return content


def chat_message_optional_content(response: Any) -> str | None:
    """Extract assistant content when vLLM emitted final content."""

    content = object_get(chat_message(response), "content")
    if isinstance(content, str) and content:
        return content
    return None


def chat_response_metadata(response: Any | None) -> dict[str, Any]:
    """Return ordinary metadata fields from a Chat Completions response."""

    if response is None:
        return {}
    choices = object_get(response, "choices") or []
    finish_reasons = [
        object_get(choice, "finish_reason")
        for choice in choices
        if object_get(choice, "finish_reason") is not None
    ]
    return {
        "response_id": object_get(response, "id"),
        "response_model": object_get(response, "model"),
        "response_object": object_get(response, "object"),
        "usage": plain(object_get(response, "usage")),
        "finish_reasons": finish_reasons,
    }
