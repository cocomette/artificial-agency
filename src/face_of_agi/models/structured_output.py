"""Shared structured-output validation and repair helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from typing import Any, Generic, TypeVar

from face_of_agi.models.providers.scheduler import current_model_call_context

ResponseT = TypeVar("ResponseT")
ValueT = TypeVar("ValueT")
DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS = 8000

OUTPUT_SCHEMA_INSTRUCTION = (
    "Output JSON must match this schema exactly. Return only the JSON response"
)


def append_output_schema_to_instructions(
    instructions: str,
    schema: dict[str, Any],
    *,
    include: bool,
) -> str:
    """Optionally append a model-readable output schema to system instructions."""

    text = instructions.strip()
    if not include:
        return text
    schema_json = json.dumps(schema, indent=2, sort_keys=True)
    return "\n\n".join((text, f"{OUTPUT_SCHEMA_INSTRUCTION}\n{schema_json}"))


@dataclass(frozen=True, slots=True)
class StructuredOutputResult(Generic[ResponseT, ValueT]):
    """Validated structured-output value and the response that produced it."""

    response: ResponseT
    value: ValueT
    repair_attempts: int


def validate_with_repair(
    *,
    label: str,
    response: ResponseT,
    text_of: Callable[[ResponseT], str],
    validate: Callable[[str], ValueT],
    repair: Callable[[str, str, int], ResponseT] | None,
    max_repair_attempts: int,
    error_factory: Callable[[str], Exception],
) -> StructuredOutputResult[ResponseT, ValueT]:
    """Validate structured output, requesting bounded repair when available."""

    repair_attempts = 0
    current_response = response
    while True:
        current_text = text_of(current_response)
        try:
            return StructuredOutputResult(
                response=current_response,
                value=validate(current_text),
                repair_attempts=repair_attempts,
            )
        except Exception as exc:
            if repair_attempts >= max_repair_attempts:
                raise error_factory(
                    f"{label} produced invalid structured output after "
                    f"{repair_attempts} repair attempt(s): {exc}; "
                    f"raw response preview: {_preview(current_text)!r}"
                ) from exc
            if repair is None:
                raise error_factory(
                    f"{label} produced invalid structured output and provider "
                    f"does not support repair: {exc}; "
                    f"raw response preview: {_preview(current_text)!r}"
                ) from exc
            repair_attempts += 1
            current_response = repair(current_text, str(exc), repair_attempts)


def provider_repair_callback(
    provider: object,
    method_name: str,
    *,
    args: Sequence[object] = (),
    kwargs: dict[str, object] | None = None,
) -> Callable[[str, str, int], ResponseT] | None:
    """Return a validate_with_repair callback for provider repair methods."""

    method = getattr(provider, method_name, None)
    if method is None:
        return None

    bound_kwargs = dict(kwargs or {})

    def repair(
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> ResponseT:
        emit_repair_attempt_event(
            provider,
            validation_error=validation_error,
            attempt=attempt,
        )
        return method(
            *args,
            **bound_kwargs,
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
        )

    return repair


def emit_repair_attempt_event(
    provider: object,
    *,
    validation_error: str,
    attempt: int,
) -> None:
    context = current_model_call_context()
    if context is None or context.emit_event is None:
        return
    provider_name = str(getattr(provider, "backend", None) or type(provider).__name__)
    model = getattr(provider, "model", None)
    context.emit_event(
        run_id=context.run_id,
        game_id=context.game_id,
        turn_id=context.turn_id,
        role=context.role,
        provider=provider_name,
        model=str(model) if model is not None else None,
        event="repair_attempt",
        status="started",
        metadata={
            "attempt": attempt,
            "validation_error_type": validation_error.split(":", 1)[0],
            "validation_error_preview": _preview(validation_error, limit=500),
        },
    )


def clipped_invalid_output_preview(
    text: str,
    *,
    max_chars: int | None = DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS,
) -> str:
    """Return a bounded head/tail preview of invalid structured output."""

    if max_chars is None:
        return text
    max_chars = int(max_chars)
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return f"[invalid output omitted: original length {len(text)} chars]"

    omitted = len(text)
    head = 0
    tail = 0
    marker = ""
    for _attempt in range(3):
        marker = f"\n\n[... omitted {omitted} chars from invalid output ...]\n\n"
        payload_budget = max_chars - len(marker)
        if payload_budget <= 0:
            return marker.strip()[:max_chars]
        head = payload_budget // 2
        tail = payload_budget - head
        omitted = len(text) - head - tail

    tail_text = text[-tail:] if tail > 0 else ""
    return text[:head] + marker + tail_text


def _preview(text: str, *, limit: int = 300) -> str:
    return text.strip().replace("\n", "\\n")[:limit]
