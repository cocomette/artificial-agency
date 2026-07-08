"""Contracts for the per-level solution summarizer role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


def level_solution_summary_json_schema() -> dict[str, Any]:
    """Return the provider-neutral level-summary output schema."""

    return {
        "type": "object",
        "properties": {
            "solution_method": {
                "type": "string",
            },
        },
        "required": ["solution_method"],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class LevelSolutionSummaryInput:
    """Input for summarizing how one completed level was solved."""

    run_id: str
    game_id: str
    completed_level: int
    strategy_history: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LevelSolutionSummary:
    """Reusable same-game method summary from one completed level."""

    solution_method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptLevelSummaryRequest:
    """Provider-neutral prompt request for a level summary."""

    instructions: str
    text: str
    output_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptLevelSummaryProviderResponse:
    """Raw provider output for one level-summary request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptLevelSummaryProvider(Protocol):
    """Thin backend boundary for one level-summary request."""

    backend: str
    model: str | None

    def summarize_level_solution(
        self,
        request: PromptLevelSummaryRequest,
    ) -> PromptLevelSummaryProviderResponse:
        """Return raw provider text for a level summary."""
        ...

    def repair_level_solution(
        self,
        request: PromptLevelSummaryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptLevelSummaryProviderResponse:
        """Return repaired raw provider text for invalid level-summary output."""
        ...


class LevelSolutionSummarizerModel(Protocol):
    """Model role that summarizes the solved method for one completed level."""

    def summarize_level_solution(
        self,
        summary_input: LevelSolutionSummaryInput,
    ) -> LevelSolutionSummary:
        """Return a compact same-game method summary."""
        ...


__all__ = [
    "LevelSolutionSummarizerModel",
    "LevelSolutionSummary",
    "LevelSolutionSummaryInput",
    "PromptLevelSummaryProvider",
    "PromptLevelSummaryProviderResponse",
    "PromptLevelSummaryRequest",
    "level_solution_summary_json_schema",
]
