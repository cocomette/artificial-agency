"""Contracts for the agent context historizer model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import ActionHistoryItem, ActionSpec, Observation
from face_of_agi.models.world.contracts import AgentContextWorldSummary

UpdaterMode = Literal["probing", "policy"]


def agent_context_history_json_schema(
    allowed_actions: tuple[ActionSpec, ...] = (),
) -> dict[str, Any]:
    """Return the provider-neutral historizer output JSON schema."""

    return {
        "type": "object",
        "properties": {
            "probing_evolution": {
                "type": "string",
            },
            "policy_evolution": {
                "type": "string",
            },
            "updater_mode": {
                "type": "string",
                "enum": ["probing", "policy"],
            },
        },
        "required": [
            "probing_evolution",
            "policy_evolution",
            "updater_mode",
        ],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class AgentContextHistoryInput:
    """Input for summarizing agent context history and selecting update mode."""

    game_id: str
    context_window: int
    strategy_history: tuple[str, ...]
    current_world_model: "AgentContextWorldSummary | None" = None
    previous_world_model: str = ""
    previous_observation: Observation | None = None
    current_observation: Observation | None = None
    action_history: tuple[ActionHistoryItem, ...] = ()
    allowed_actions: tuple[ActionSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContextHistoryDecision:
    """Historizer output fields and selected update mode."""

    probing_evolution: str
    policy_evolution: str
    updater_mode: UpdaterMode
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContextHistorySummary:
    """Summary of how prior agent game-context fields evolved."""

    world_description: str
    action_effects: dict[str, str]
    updater_mode: UpdaterMode
    probing_evolution: str = ""
    policy_evolution: str = ""
    special_events: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_available(cls) -> "AgentContextHistorySummary":
        """Return an explicit summary for unavailable context history."""

        return cls(
            world_description="not available",
            probing_evolution="not available",
            policy_evolution="not available",
            action_effects={},
            special_events="not available",
            updater_mode="probing",
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
    """Model role that summarizes agent-context history."""

    def summarize_agent_context_history(
        self,
        history_input: AgentContextHistoryInput,
    ) -> AgentContextHistorySummary:
        """Return a history summary and next updater mode."""
        ...
