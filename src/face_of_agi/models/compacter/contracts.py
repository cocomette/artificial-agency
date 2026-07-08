"""Contracts for the compacter model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import ActionHistoryItem, ActionSpec, Observation


def agent_compacter_json_schema(
    allowed_actions: tuple[ActionSpec, ...] = (),
) -> dict[str, Any]:
    """Return the compacter output JSON schema."""

    action_effect_properties = {
        action.name: {
            "type": "string",
        }
        for action in allowed_actions
    }
    action_effect_schema: dict[str, Any] = {
        "type": "object",
        "properties": action_effect_properties,
        "additionalProperties": False if allowed_actions else {"type": "string"},
    }
    if allowed_actions:
        action_effect_schema["required"] = list(action_effect_properties)
    return {
        "type": "object",
        "properties": {
            "world_description": {
                "type": "string",
            },
            "special_events": {
                "type": "string",
            },
            "action_effects": action_effect_schema,
            "previous_actions_summary": {
                "type": "string",
            },
            "previous_strategy_summary": {
                "type": "string",
            },
        },
        "required": [
            "world_description",
            "special_events",
            "action_effects",
            "previous_actions_summary",
            "previous_strategy_summary",
        ],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class AgentCompacterInput:
    """Input for compacting current game context."""

    game_id: str
    current_observation: Observation
    previous_compacter_context: str = ""
    action_history: tuple[ActionHistoryItem, ...] = ()
    strategy_history: tuple[str, ...] = ()
    allowed_actions: tuple[ActionSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCompacterSummary:
    """Compact current game context for later model calls."""

    world_description: str
    action_effects: dict[str, str]
    previous_actions_summary: str = ""
    previous_strategy_summary: str = ""
    special_events: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptCompacterRequest:
    """Provider-neutral prompt request for the compacter role."""

    instructions: str
    text: str
    output_schema: dict[str, Any]
    images: tuple["PromptCompacterImage", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptCompacterImage:
    """Provider-neutral image attached to a compacter request."""

    label: str
    image: Any


@dataclass(slots=True)
class PromptCompacterProviderResponse:
    """Raw provider output for one compacter request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptCompacterProvider(Protocol):
    """Thin backend boundary for one compacter request."""

    backend: str
    model: str | None

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        """Return raw provider text for a compacter summary."""
        ...

    def repair_compacter_context(
        self,
        request: PromptCompacterRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptCompacterProviderResponse:
        """Return repaired raw provider text for invalid compacter output."""
        ...


class AgentCompacterModel(Protocol):
    """Model role that compacts current game context."""

    def compact_agent_context(
        self,
        compacter_input: AgentCompacterInput,
    ) -> AgentCompacterSummary:
        """Return a compact context summary for one frame turn."""
        ...


__all__ = [
    "AgentCompacterInput",
    "AgentCompacterModel",
    "AgentCompacterSummary",
    "PromptCompacterImage",
    "PromptCompacterProvider",
    "PromptCompacterProviderResponse",
    "PromptCompacterRequest",
    "agent_compacter_json_schema",
]
