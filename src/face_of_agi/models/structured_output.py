"""Shared structured-output validation and repair helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from typing import Any, Generic, TypeVar

ResponseT = TypeVar("ResponseT")
ValueT = TypeVar("ValueT")

OUTPUT_SCHEMA_INSTRUCTION = (
    "Output JSON must match this schema exactly. Return only the JSON response"
)
MODEL_FALLBACK_WARNING = (
    "max repair attempts / model context length reached, continuing with "
    "%s fallback backend=%s model=%s repair_attempts=%s reason=%s"
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
        return method(
            *args,
            **bound_kwargs,
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
        )

    return repair


def readable_model_error(exc: Exception, *, limit: int = 500) -> str:
    """Return a concise single-line provider/model error for operator logs."""

    message = str(exc).strip().replace("\n", " ")
    if not message:
        message = type(exc).__name__
    return f"{type(exc).__name__}: {message}"[:limit]


def _preview(text: str, *, limit: int = 300) -> str:
    return text.strip().replace("\n", "\\n")[:limit]
