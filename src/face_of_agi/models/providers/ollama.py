"""Final Ollama chat provider-call helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

_FORMAT_UNSET = object()
OLLAMA_JSON_ASSISTANT_PREFILL = "```json\n"


def structured_json_instructions(instructions: str, schema: dict[str, Any]) -> str:
    """Add Ollama-specific JSON contract text to system instructions."""

    return "\n\n".join(
        [
            instructions,
            "Output contract: return exactly one JSON value and no explanatory "
            "text. The JSON must validate against this exact schema:",
            json.dumps(schema, indent=2, sort_keys=True),
        ]
    )


def assistant_json_prefill_message() -> dict[str, str]:
    """Return the Ollama assistant prefill used for JSON generations."""

    return {"role": "assistant", "content": OLLAMA_JSON_ASSISTANT_PREFILL}


@dataclass(slots=True)
class OllamaChatConfig:
    """Minimal shared config for direct Ollama chat calls."""

    model: str | None = None
    host: str | None = None
    think: bool | str | None = False
    format: str | dict[str, Any] | None = None
    keep_alive: int | str | None = None
    options: dict[str, Any] = field(default_factory=dict)


class OllamaChatClient:
    """Last-step Ollama chat caller for role-specific adapters."""

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
        response_format: Any = _FORMAT_UNSET,
    ) -> Any:
        """Build and send the final Ollama chat request."""

        request = self.build_request(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
        )
        self.last_request = request
        return self._require_client().chat(**request)

    def build_request(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: Any = _FORMAT_UNSET,
    ) -> dict[str, Any]:
        """Build the final Ollama chat request without sending it."""

        if not model:
            raise ValueError("Ollama chat calls require an explicit model")
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools is not None:
            request["tools"] = tools
        think = getattr(self.config, "think", None)
        if think is not None:
            request["think"] = think
        options = getattr(self.config, "options", None)
        if options:
            request["options"] = options
        resolved_format = (
            getattr(self.config, "format", None)
            if response_format is _FORMAT_UNSET
            else response_format
        )
        if resolved_format is not None:
            request["format"] = resolved_format
        keep_alive = getattr(self.config, "keep_alive", None)
        if keep_alive is not None:
            request["keep_alive"] = keep_alive
        return request

    def _require_client(self) -> Any:
        if self._client is None:
            import ollama

            host = getattr(self.config, "host", None)
            if host:
                self._client = ollama.Client(host=host)
            else:
                self._client = ollama
        return self._client


def object_get(value: Any, key: str, default: Any = None) -> Any:
    """Read a key from SDK objects, dictionaries, or test doubles."""

    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def response_usage(response: Any) -> dict[str, Any]:
    """Return normalized Ollama usage/timing metadata."""

    keys = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "done_reason",
    )
    return {
        key: object_get(response, key)
        for key in keys
        if object_get(response, key) is not None
    }


def message_content(response: Any) -> str:
    """Extract required assistant message content from an Ollama response."""

    message = object_get(response, "message") or {}
    content = object_get(message, "content")
    if not isinstance(content, str) or content == "":
        raise RuntimeError("Ollama text response did not include message content")
    return content


def structured_json_content(response: Any) -> str:
    """Extract JSON text from an Ollama structured-output response."""

    return _strip_json_fence(message_content(response).strip())


def _strip_json_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text
