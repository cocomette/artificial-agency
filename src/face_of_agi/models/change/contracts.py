"""Contracts for transition change summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from face_of_agi.contracts import ActionSpec, Observation

DEFAULT_CHANGE_SUMMARY_MAX_CHARS = 2000


def change_summary_json_schema(
    *,
    summary_max_chars: int | None = DEFAULT_CHANGE_SUMMARY_MAX_CHARS,
) -> dict[str, Any]:
    """Return the provider-neutral change-summary output schema."""

    summary_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "description": "One or two concise sentences describing visible change.",
    }
    if summary_max_chars is not None:
        summary_schema["maxLength"] = int(summary_max_chars)

    return {
        "type": "object",
        "properties": {
            "summary": summary_schema,
            "change_detected": {
                "type": "boolean",
                "description": (
                    "Whether any adjacent serialized frame pair changed inside "
                    "the cropped ARC-grid area."
                ),
            },
        },
        "required": ["summary", "change_detected"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class ChangeSummaryResult:
    """Provider-neutral output from the change summary role."""

    summary: str
    changed_pixel_count: int
    change_detected: bool
    metadata: dict[str, Any]
    changed_cell_percent: float | None = None


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
        images: Sequence[Any],
        output_schema: dict[str, Any],
    ) -> ChangeSummaryProviderResponse:
        """Return raw provider text for a transition change summary."""
        ...

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> ChangeSummaryProviderResponse:
        """Return repaired provider text for invalid structured output."""
        ...

    def reduce_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
    ) -> ChangeSummaryProviderResponse:
        """Return raw provider text for a final reduced change summary."""
        ...

    def repair_reduce_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> ChangeSummaryProviderResponse:
        """Return repaired provider text for an invalid reduced summary."""
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
    ) -> ChangeSummaryResult:
        """Return one compact visual change summary."""
        ...
