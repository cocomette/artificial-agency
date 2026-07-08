"""Contracts for the historizer model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import ActionHistoryItem


def historizer_summary_json_schema() -> dict[str, Any]:
    """Return the provider-neutral historizer output schema."""

    return {
        "type": "object",
        "properties": {
            "action_history_summary": {"type": "string"},
            "strategy_history_summary": {"type": "string"},
        },
        "required": ["action_history_summary", "strategy_history_summary"],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class HistorizerInput:
    """Input for compacting current-level history for updater P."""

    run_id: str
    game_id: str
    action_history: tuple[ActionHistoryItem, ...]
    strategy_history: tuple[str, ...]
    world_model_context: str = ""
    previous_history_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HistorizerSummary:
    """Compacted current-level action and strategy history."""

    action_history_summary: str
    strategy_history_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptHistorizerRequest:
    """Provider-neutral prompt request for historizer calls."""

    instructions: str
    text: str
    output_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptHistorizerProviderResponse:
    """Raw provider output for one historizer request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptHistorizerProvider(Protocol):
    """Thin backend boundary for one historizer request."""

    backend: str
    model: str | None

    def summarize_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        """Return raw provider text for a historizer summary."""
        ...

    def repair_history(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptHistorizerProviderResponse:
        """Return repaired raw provider text for invalid historizer output."""
        ...


class HistorizerModel(Protocol):
    """Model role that compacts action and strategy history."""

    def summarize_history(
        self,
        historizer_input: HistorizerInput,
    ) -> HistorizerSummary:
        """Return compact current-level history summaries."""
        ...


__all__ = [
    "HistorizerInput",
    "HistorizerModel",
    "HistorizerSummary",
    "PromptHistorizerProvider",
    "PromptHistorizerProviderResponse",
    "PromptHistorizerRequest",
    "historizer_summary_json_schema",
]
