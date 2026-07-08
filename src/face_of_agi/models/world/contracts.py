"""Contracts for the agent world-model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import ActionHistoryItem, ActionSpec, Observation


def agent_world_model_json_schema(
    allowed_actions: tuple[ActionSpec, ...] = (),
) -> dict[str, Any]:
    """Return the world-model output JSON schema."""

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
        },
        "required": [
            "world_description",
            "special_events",
            "action_effects",
        ],
        "additionalProperties": False,
    }


@dataclass(slots=True)
class AgentWorldModelInput:
    """Input for summarizing current game-world mechanics."""

    game_id: str
    current_observation: Observation
    previous_world_model: str = ""
    action_history: tuple[ActionHistoryItem, ...] = ()
    allowed_actions: tuple[ActionSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContextWorldSummary:
    """World-model summary of current mechanics and action effects."""

    world_description: str
    action_effects: dict[str, str]
    special_events: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptWorldRequest:
    """Provider-neutral prompt request for the world-model role."""

    instructions: str
    text: str
    output_schema: dict[str, Any]
    images: tuple["PromptWorldImage", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptWorldImage:
    """Provider-neutral image attached to a world-model request."""

    label: str
    image: Any


@dataclass(slots=True)
class PromptWorldProviderResponse:
    """Raw provider output for one world-model request."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptWorldProvider(Protocol):
    """Thin backend boundary for one world-model request."""

    backend: str
    model: str | None

    def summarize_world_model(
        self,
        request: PromptWorldRequest,
    ) -> PromptWorldProviderResponse:
        """Return raw provider text for a world-model summary."""
        ...

    def repair_world_model(
        self,
        request: PromptWorldRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptWorldProviderResponse:
        """Return repaired raw provider text for invalid world-model output."""
        ...


class AgentWorldModel(Protocol):
    """Model role that summarizes world description and action effects."""

    def summarize_agent_world_model(
        self,
        world_input: AgentWorldModelInput,
    ) -> AgentContextWorldSummary:
        """Return a world-model summary for one frame transition."""
        ...


__all__ = [
    "AgentContextWorldSummary",
    "AgentWorldModel",
    "AgentWorldModelInput",
    "PromptWorldImage",
    "PromptWorldProvider",
    "PromptWorldProviderResponse",
    "PromptWorldRequest",
    "agent_world_model_json_schema",
]
