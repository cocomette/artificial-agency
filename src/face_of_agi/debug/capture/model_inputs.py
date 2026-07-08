"""Provider-input capture helpers for debug-only inspection."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_RECORDS_ATTR = "_model_input_debug_records"


def capture_model_input(
    target: object,
    *,
    call_slot: str,
    provider: str,
    model: str | None,
    phase: str,
    request: dict[str, Any],
    usage: Any | None = None,
    metadata: dict[str, Any] | None = None,
    attempt: int | None = None,
) -> None:
    """Append one raw provider request record to an adapter/provider object."""

    records = _records(target)
    records.append(
        {
            "call_slot": call_slot,
            "provider": provider,
            "model": model,
            "phase": phase,
            "attempt": (
                _next_attempt(records, call_slot=call_slot, phase=phase)
                if attempt is None
                else attempt
            ),
            "request": _jsonable(request),
            "usage": _jsonable(usage),
            "metadata": _jsonable(metadata or {}),
        }
    )


def capture_openai_model_input(
    target: object,
    *,
    call_slot: str,
    provider: str,
    model: str | None,
    phase: str,
    request: dict[str, Any],
    response: Any | None,
    metadata: dict[str, Any] | None = None,
    attempt: int | None = None,
) -> None:
    """Capture a raw OpenAI-style request plus generic response metadata."""

    response_metadata = _response_usage_metadata(response)
    capture_model_input(
        target,
        call_slot=call_slot,
        provider=provider,
        model=model,
        phase=phase,
        attempt=attempt,
        request=request,
        usage=response_metadata.get("usage"),
        metadata={
            "backend": provider,
            "model": model,
            **response_metadata,
            "response_output_text": _response_text(response),
            "response_metadata": response_metadata,
            "response_payload": _jsonable(response),
            **(metadata or {}),
        },
    )


def capture_ollama_model_input(
    target: object,
    *,
    call_slot: str,
    provider: str,
    model: str | None,
    phase: str,
    request: dict[str, Any],
    response: Any | None,
    metadata: dict[str, Any] | None = None,
    attempt: int | None = None,
) -> None:
    """Capture a raw Ollama-style chat request plus generic response metadata."""

    response_metadata = _response_usage_metadata(response)
    capture_model_input(
        target,
        call_slot=call_slot,
        provider=provider,
        model=model,
        phase=phase,
        attempt=attempt,
        request=request,
        usage=response_metadata.get("usage"),
        metadata={
            "backend": provider,
            "model": model,
            "response_output_text": _ollama_response_output_text(response),
            "response_metadata": response_metadata,
            "response_payload": _jsonable(response),
            **(metadata or {}),
        },
    )


def capture_chat_model_input(
    target: object,
    *,
    call_slot: str,
    provider: str,
    model: str | None,
    phase: str,
    request: dict[str, Any],
    response: Any | None,
    metadata: dict[str, Any] | None = None,
    attempt: int | None = None,
) -> None:
    """Capture a raw OpenAI-compatible chat request plus generic metadata."""

    response_metadata = _response_usage_metadata(response)
    capture_model_input(
        target,
        call_slot=call_slot,
        provider=provider,
        model=model,
        phase=phase,
        attempt=attempt,
        request=request,
        usage=response_metadata.get("usage"),
        metadata={
            "backend": provider,
            "model": model,
            **response_metadata,
            "response_output_text": _response_text(response),
            "response_metadata": response_metadata,
            "response_payload": _jsonable(response),
            **(metadata or {}),
        },
    )


def drain_model_input_debug_records(source: object) -> list[dict[str, Any]]:
    """Collect and clear captured provider-input records from an adapter tree."""

    drained: list[dict[str, Any]] = []
    seen: set[int] = set()
    for target in _candidate_targets(source):
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)

        records = getattr(target, _RECORDS_ATTR, None)
        if not records:
            continue
        drained.extend(records)
        setattr(target, _RECORDS_ATTR, [])
    return drained


def _candidate_targets(source: object) -> tuple[object, ...]:
    candidates: list[object] = [source]
    for attr in ("provider", "_provider"):
        nested = getattr(source, attr, None)
        if nested is not None:
            candidates.append(nested)
    return tuple(candidates)


def _records(target: object) -> list[dict[str, Any]]:
    records = getattr(target, _RECORDS_ATTR, None)
    if records is None:
        records = []
        setattr(target, _RECORDS_ATTR, records)
    return records


def _next_attempt(
    records: list[dict[str, Any]],
    *,
    call_slot: str,
    phase: str,
) -> int:
    return sum(
        1
        for record in records
        if record.get("call_slot") == call_slot and record.get("phase") == phase
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json", exclude_none=True))
        except TypeError:
            return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


def _ollama_response_output_text(response: Any | None) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        message = response.get("message") or {}
    else:
        message = getattr(response, "message", {}) or {}
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return None


def _response_usage_metadata(response: Any | None) -> dict[str, Any]:
    if response is None:
        return {}
    usage = getattr(response, "usage", None)
    return {"usage": _jsonable(usage)} if usage is not None else {}


def _response_text(response: Any | None) -> str | None:
    if response is None:
        return None
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
    return None
