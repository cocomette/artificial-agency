"""Contracts for transition change summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol

from face_of_agi.contracts import ActionSpec, Observation


def change_summary_json_schema() -> dict[str, Any]:
    """Return the provider-neutral change-summary output schema."""

    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One or two concise sentences describing visible change.",
            },
            "change_detected": {
                "type": "boolean",
                "description": "Whether the attached frames show a visible change.",
            },
        },
        "required": ["summary", "change_detected"],
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

    summary: str
    changed_pixel_percent: float
    change_detected: bool
    metadata: dict[str, Any]


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
        changed_pixel_percent: float,
        frame_observations: Sequence[Observation] | None = None,
        max_transition_changed_pixel_percent: float | None = None,
    ) -> ChangeSummaryResult:
        """Return one compact visual change summary."""
        ...
