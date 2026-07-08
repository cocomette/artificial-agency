"""Final Ollama chat provider-call helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_FORMAT_UNSET = object()
OLLAMA_JSON_ASSISTANT_PREFILL = "```json\n"
OLLAMA_JSON_CONVERSION_INSTRUCTIONS = (
    "Convert the previous assistant response into strict JSON for the current "
    "task. Return only the JSON object or array requested by the schema. Do not "
    "include markdown, prose, comments, or placeholders."
)


def assistant_json_prefill_message() -> dict[str, str]:
    """Return the Ollama assistant prefill used for JSON generations."""

    return {"role": "assistant", "content": OLLAMA_JSON_ASSISTANT_PREFILL}


@dataclass(slots=True)
class OllamaChatConfig:
    """Minimal shared config for direct Ollama chat calls."""

    model: str | None = None
    host: str | None = None
    think: bool | str | None = None
    format: str | dict[str, Any] | None = None
    keep_alive: int | str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OllamaChatCall:
    """One raw Ollama request/response pair from a structured role call."""

    kind: str
    request: dict[str, Any]
    response: Any


@dataclass(frozen=True, slots=True)
class OllamaStructuredChatResult:
    """Structured call result plus debug-visible intermediate calls."""

    response: Any
    calls: tuple[OllamaChatCall, ...]

    @property
    def request(self) -> dict[str, Any]:
        """Return the request that produced the returned response."""

        return self.calls[-1].request


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

    def structured_chat(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        response_format: Any,
        tools: list[dict[str, Any]] | None = None,
    ) -> OllamaStructuredChatResult:
        """Call Ollama for structured output, preserving thinking when enabled."""

        if not _two_pass_thinking_enabled(self.config):
            response = self.chat(
                model=model,
                messages=messages,
                tools=tools,
                response_format=response_format,
            )
            if self.last_request is None:
                raise RuntimeError("Ollama chat call did not record its request")
            return OllamaStructuredChatResult(
                response=response,
                calls=(
                    OllamaChatCall(
                        kind="structured",
                        request=self.last_request,
                        response=response,
                    ),
                ),
            )

        thinking_request = self.build_request(
            model=model,
            messages=_without_json_prefill(messages),
            tools=tools,
            response_format=None,
        )
        self.last_request = thinking_request
        thinking_response = self._require_client().chat(**thinking_request)
        thinking_call = OllamaChatCall(
            kind="thinking",
            request=thinking_request,
            response=thinking_response,
        )
        if _response_tool_calls(thinking_response):
            return OllamaStructuredChatResult(
                response=thinking_response,
                calls=(thinking_call,),
            )

        structured_request = self.build_request(
            model=model,
            messages=_conversion_messages(thinking_response),
            tools=None,
            response_format=response_format,
            think=False,
        )
        self.last_request = structured_request
        structured_response = self._require_client().chat(**structured_request)
        structured_call = OllamaChatCall(
            kind="structured",
            request=structured_request,
            response=structured_response,
        )
        return OllamaStructuredChatResult(
            response=structured_response,
            calls=(thinking_call, structured_call),
        )

    def build_request(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: Any = _FORMAT_UNSET,
        think: Any = _FORMAT_UNSET,
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
        resolved_think = (
            getattr(self.config, "think", None)
            if think is _FORMAT_UNSET
            else think
        )
        if resolved_think is not None:
            request["think"] = resolved_think
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


def _two_pass_thinking_enabled(config: Any) -> bool:
    return getattr(config, "think", None) is True


def _without_json_prefill(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    last = messages[-1]
    if (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and last.get("content") == OLLAMA_JSON_ASSISTANT_PREFILL
    ):
        return list(messages[:-1])
    return list(messages)


def _conversion_messages(response: Any) -> list[dict[str, str]]:
    message = object_get(response, "message") or {}
    thinking = object_get(message, "thinking")
    content = object_get(message, "content")
    parts = []
    if isinstance(thinking, str) and thinking.strip():
        parts.append("Previous assistant thinking trace:\n" + thinking.strip())
    if isinstance(content, str) and content.strip():
        parts.append("Previous assistant answer:\n" + content.strip())
    if not parts:
        parts.append("Previous assistant answer:\n")
    return [
        {"role": "system", "content": OLLAMA_JSON_CONVERSION_INSTRUCTIONS},
        {"role": "user", "content": "\n\n".join(parts)},
        assistant_json_prefill_message(),
    ]


def _response_tool_calls(response: Any) -> bool:
    message = object_get(response, "message") or {}
    return bool(object_get(message, "tool_calls", None))


def _strip_json_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text
