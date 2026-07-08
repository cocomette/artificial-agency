"""Contracts for transition change summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol

from face_of_agi.contracts import ActionSpec, ChangeSummaryElement, Observation

DEFAULT_CHANGE_SUMMARY_MAX_CHARS = 2_000
DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS = 20


def change_summary_json_schema(
    *,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
    summary_max_elements: int | None = DEFAULT_CHANGE_SUMMARY_MAX_ELEMENTS,
) -> dict[str, Any]:
    """Return the provider-neutral change-summary output schema."""

    field_schema: dict[str, Any] = {"type": "string"}
    if summary_max_chars is not None:
        field_schema["maxLength"] = int(summary_max_chars)

    return {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "description": "Visible elements and their chronological mutations.",
                **(
                    {"maxItems": int(summary_max_elements)}
                    if summary_max_elements is not None
                    else {}
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "element_name": {
                            **field_schema,
                            "minLength": 1,
                        },
                        "element_description": {
                            **field_schema,
                            "minLength": 1,
                        },
                        "element_mutation": field_schema,
                    },
                    "required": [
                        "element_name",
                        "element_description",
                        "element_mutation",
                    ],
                    "additionalProperties": False,
                },
            },
            "change_detected": {
                "type": "boolean",
                "description": (
                    "Whether any adjacent attached frame pair changed inside "
                    "the model-visible area."
                ),
            },
        },
        "required": ["elements", "change_detected"],
        "additionalProperties": False,
    }


def openai_change_summary_text_format(
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return OpenAI Responses text format for change summaries."""

    return {
        "format": {
            "type": "json_schema",
            "name": "change_summary",
            "strict": True,
            "schema": schema or change_summary_json_schema(),
        }
    }


@dataclass(frozen=True, slots=True)
class ChangeSummaryResult:
    """Provider-neutral output from the change summary role."""

    elements: tuple[ChangeSummaryElement, ...]
    changed_pixel_count: int
    change_detected: bool
    metadata: dict[str, Any]
    changed_pixel_percent: float | None = None


@dataclass(frozen=True, slots=True)
class ChangeSummaryProviderResponse:
    """Raw provider output plus diagnostics for a change summary call."""

    text: str
    metadata: dict[str, Any]
    request: dict[str, Any] | None = None


class ChangeSummaryProvider(Protocol):
    """Provider transport for transition change summary prompts."""

    backend: str
    model: str | None

    def complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        previous_image: Any,
        current_image: Any,
        output_schema: dict[str, Any],
        images: Sequence[Any] | None = None,
    ) -> ChangeSummaryProviderResponse:
        """Return raw provider text for a transition change summary."""
        ...

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        previous_image: Any,
        current_image: Any,
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
        images: Sequence[Any] | None = None,
    ) -> ChangeSummaryProviderResponse:
        """Return repaired provider text for invalid structured output."""
        ...


class ChangeSummaryModel(Protocol):
    """Model role that summarizes visual change between two frames."""

    def summarize(
        self,
        previous_observation: Observation,
        current_observation: Observation,
        action: ActionSpec,
        *,
        glossary_actions: Sequence[ActionSpec],
        frame_observations: Sequence[Observation] | None = None,
        previous_change_elements: Sequence[ChangeSummaryElement] = (),
    ) -> ChangeSummaryResult:
        """Return one compact visual change summary."""
        ...
