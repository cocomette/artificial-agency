"""Contracts for the agent context historizer model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

AGENT_CONTEXT_HISTORY_KEYS = (
    "goals",
    "game_mechanics",
    "policy",
    "history",
    "extras",
)


def agent_context_history_json_schema() -> dict[str, Any]:
    """Return the provider-neutral historizer output JSON schema."""

    descriptions = {
        "goals": "How objective and goal hypotheses changed over the context history.",
        "game_mechanics": "How mechanics/action-effect beliefs changed over time.",
        "policy": "How action-selection guidance changed over time.",
        "history": "How learned outcome/progress lessons changed over time.",
        "extras": "How miscellaneous guidance changed over time.",
    }
    return {
        "type": "object",
        "properties": {
            "field_evolution": {
                "type": "object",
                "description": (
                    "Summary of how each agent game-context field evolved "
                    "across the provided oldest-to-newest context history."
                ),
                "properties": {
                    key: {
                        "type": "string",
                        "description": descriptions[key],
                    }
                    for key in AGENT_CONTEXT_HISTORY_KEYS
                },
                "required": list(AGENT_CONTEXT_HISTORY_KEYS),
                "additionalProperties": False,
            },
        },
        "required": ["field_evolution"],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class AgentContextHistoryInput:
    """Input for summarizing prior agent game-context evolution."""

    game_id: str
    context_window: int
    contexts: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContextHistorySummary:
    """Summary of how prior agent game-context fields evolved."""

    field_evolution: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_available(cls) -> "AgentContextHistorySummary":
        """Return an explicit summary for unavailable context history."""

        return cls(
            field_evolution={
                key: "not available" for key in AGENT_CONTEXT_HISTORY_KEYS
            },
            metadata={"available": False},
        )

    def is_available(self) -> bool:
        """Return whether this summary came from prior context history."""

        return bool(self.metadata.get("available", True))


@dataclass(slots=True)
class PromptHistorizerRequest:
    """Provider-neutral prompt request for the historizer role."""

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

    def summarize_context_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        """Return raw provider text for a context-history summary."""
        ...

    def repair_context_history(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptHistorizerProviderResponse:
        """Return repaired raw provider text for invalid structured output."""
        ...


class AgentContextHistorizerModel(Protocol):
    """Model role that summarizes prior agent-context field evolution."""

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        """Return a field-evolution summary."""
        ...
