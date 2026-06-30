"""Final vLLM OpenAI-compatible chat provider-call helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_API_KEY_ENV = "VLLM_API_KEY"
DEFAULT_CONTEXT_TRUNCATION_MARGIN_TOKENS = 256
DEFAULT_CONTEXT_OVERFLOW_RETRIES = 3
TRUNCATION_MARKER = "\n\n[... truncated to fit vLLM context window ...]\n\n"
_OVERFLOW_RE = re.compile(
    r"maximum context length is (?P<max>\d+) tokens.*?"
    r"requested (?P<output>\d+) output tokens.*?"
    r"prompt contains at least (?P<input>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


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


def set_optional(target: dict[str, Any], key: str, value: Any) -> None:
    """Set a request field when it carries a meaningful value."""

    if value is None:
        return
    if value == {} or value == []:
        return
    target[key] = value


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
    max_context_tokens: int | None = None
    truncate_context_on_overflow: bool = True
    context_truncation_margin_tokens: int = DEFAULT_CONTEXT_TRUNCATION_MARGIN_TOKENS
    context_overflow_retries: int = DEFAULT_CONTEXT_OVERFLOW_RETRIES
    options: dict[str, Any] = field(default_factory=dict)
    extra_request_options: dict[str, Any] = field(default_factory=dict)


class VLLMChatClient:
    """Last-step vLLM Chat Completions caller for role-specific adapters."""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client
        self.last_request: dict[str, Any] | None = None
        self.last_truncation: dict[str, Any] | None = None

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
        self.last_truncation = None
        max_attempts = max(0, int(getattr(self.config, "context_overflow_retries", 0)))
        for attempt in range(max_attempts + 1):
            self.last_request = request
            try:
                return self._require_client().chat.completions.create(**request)
            except Exception as exc:
                overflow = _context_overflow_from_exception(exc)
                if (
                    overflow is None
                    or attempt >= max_attempts
                    or not getattr(self.config, "truncate_context_on_overflow", True)
                ):
                    raise
                request = self._truncate_overflowing_request(
                    request,
                    overflow=overflow,
                    attempt=attempt + 1,
                )

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

    def _truncate_overflowing_request(
        self,
        request: dict[str, Any],
        *,
        overflow: "ContextOverflow",
        attempt: int,
    ) -> dict[str, Any]:
        max_context_tokens = _configured_max_context_tokens(self.config, overflow)
        output_tokens = max(
            overflow.requested_output_tokens,
            _configured_output_tokens(self.config),
        )
        margin = max(
            0,
            int(
                getattr(
                    self.config,
                    "context_truncation_margin_tokens",
                    DEFAULT_CONTEXT_TRUNCATION_MARGIN_TOKENS,
                )
            ),
        )
        prompt_token_budget = max(1, max_context_tokens - output_tokens - margin)
        target_ratio = min(0.95, prompt_token_budget / overflow.input_tokens)
        truncated_request, truncation = truncate_chat_request_messages(
            request,
            target_ratio=target_ratio,
        )
        if truncation["removed_chars"] <= 0:
            raise RuntimeError(
                "vLLM context overflow could not be recovered because no "
                "mutable chat message text was available to truncate"
            )
        self.last_truncation = {
            **truncation,
            "attempt": attempt,
            "max_context_tokens": max_context_tokens,
            "prompt_token_budget": prompt_token_budget,
            "reported_input_tokens": overflow.input_tokens,
            "reported_requested_output_tokens": overflow.requested_output_tokens,
            "configured_output_tokens": output_tokens,
        }
        return truncated_request


@dataclass(frozen=True, slots=True)
class ContextOverflow:
    """Parsed vLLM context overflow details."""

    max_context_tokens: int
    requested_output_tokens: int
    input_tokens: int


def _context_overflow_from_exception(exc: Exception) -> ContextOverflow | None:
    """Return context overflow details parsed from a vLLM/OpenAI error."""

    match = _OVERFLOW_RE.search(str(exc))
    if match is None:
        return None
    return ContextOverflow(
        max_context_tokens=int(match.group("max")),
        requested_output_tokens=int(match.group("output")),
        input_tokens=int(match.group("input")),
    )


def _configured_max_context_tokens(config: Any, overflow: ContextOverflow) -> int:
    value = getattr(config, "max_context_tokens", None)
    if value is None:
        value = _config_options(config).get("max_context_tokens")
    if value is None:
        return overflow.max_context_tokens
    return int(value)


def _configured_output_tokens(config: Any) -> int:
    for source in (config, _config_options(config)):
        for key in ("max_completion_tokens", "max_tokens"):
            if isinstance(source, dict):
                value = source.get(key)
            else:
                value = getattr(source, key, None)
            if value is not None:
                return max(0, int(value))
    return 0


def _config_options(config: Any) -> dict[str, Any]:
    options = getattr(config, "options", None)
    return options if isinstance(options, dict) else {}


def truncate_chat_request_messages(
    request: dict[str, Any],
    *,
    target_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a request with mutable message text shortened by ratio."""

    messages = request.get("messages")
    if not isinstance(messages, list):
        return dict(request), {
            "removed_chars": 0,
            "original_chars": 0,
            "target_chars": 0,
        }

    mutable_lengths = [
        _mutable_text_length(message.get("content"))
        for message in messages
        if _is_mutable_text_message(message)
    ]
    original_chars = sum(mutable_lengths)
    target_chars = max(0, int(original_chars * max(0.0, min(1.0, target_ratio))))
    if original_chars <= target_chars:
        return dict(request), {
            "removed_chars": 0,
            "original_chars": original_chars,
            "target_chars": target_chars,
        }

    allocations = _proportional_allocations(mutable_lengths, target_chars)
    allocation_index = 0
    truncated_messages: list[Any] = []
    removed_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            truncated_messages.append(message)
            continue
        copied_message = dict(message)
        if _is_mutable_text_message(message):
            content = message["content"]
            max_chars = allocations[allocation_index]
            allocation_index += 1
            copied_content, removed = _truncate_message_content(content, max_chars)
            copied_message["content"] = copied_content
            removed_chars += removed
        truncated_messages.append(copied_message)

    truncated_request = dict(request)
    truncated_request["messages"] = truncated_messages
    return truncated_request, {
        "removed_chars": removed_chars,
        "original_chars": original_chars,
        "target_chars": target_chars,
    }


def _is_mutable_text_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") in {"system", "developer"}:
        return False
    return _mutable_text_length(message.get("content")) > 0


def _mutable_text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part.get("text"))
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return 0


def _truncate_message_content(content: Any, max_chars: int) -> tuple[Any, int]:
    if isinstance(content, str):
        truncated = _truncate_text_middle(content, max_chars)
        return truncated, len(content) - len(truncated)
    if not isinstance(content, list):
        return content, 0

    text_lengths = [
        len(part.get("text"))
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ]
    allocations = _proportional_allocations(text_lengths, max_chars)
    allocation_index = 0
    copied_parts: list[Any] = []
    removed = 0
    for part in content:
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ):
            copied_part = dict(part)
            text = part["text"]
            allocated = allocations[allocation_index]
            allocation_index += 1
            copied_part["text"] = _truncate_text_middle(text, allocated)
            removed += len(text) - len(copied_part["text"])
            copied_parts.append(copied_part)
        else:
            copied_parts.append(part)
    return copied_parts, removed


def _proportional_allocations(lengths: list[int], target_total: int) -> list[int]:
    if not lengths:
        return []
    total = sum(lengths)
    if total <= 0:
        return [0 for _length in lengths]
    allocations: list[int] = []
    remaining = target_total
    for index, length in enumerate(lengths):
        if index == len(lengths) - 1:
            allocation = max(0, min(length, remaining))
        else:
            allocation = max(0, min(length, int(target_total * (length / total))))
        allocations.append(allocation)
        remaining -= allocation
    return allocations


def _truncate_text_middle(text: str, max_chars: int) -> str:
    if max_chars >= len(text):
        return text
    if max_chars <= 0:
        return ""
    if max_chars <= len(TRUNCATION_MARKER) + 2:
        return text[-max_chars:]
    payload_chars = max_chars - len(TRUNCATION_MARKER)
    head_chars = payload_chars // 2
    tail_chars = payload_chars - head_chars
    return text[:head_chars] + TRUNCATION_MARKER + text[-tail_chars:]


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
