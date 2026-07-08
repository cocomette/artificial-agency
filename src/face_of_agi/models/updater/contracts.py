"""Contracts for the updater model P."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionSpec,
    Observation,
    RoleContext,
)

UpdaterRole = Literal["agent"]
ContextSegment = Literal["game"]
UpdaterTask = Literal["agent"]
AGENT_GAME_CONTEXT_KEYS = (
    "current_strategy",
)
AGENT_GAME_CONTEXT_MAX_CHARS = 6000


def updated_context_json_schema() -> dict[str, Any]:
    """Return the provider-neutral updater output JSON schema."""

    return {
        "type": "object",
        "properties": {
            "updated_context": {
                "type": "string",
                "description": "The complete revised context text.",
            },
        },
        "required": ["updated_context"],
        "additionalProperties": False,
    }


def _action_output_schema(allowed_actions: Sequence[ActionSpec]) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    for action in allowed_actions:
        properties: dict[str, Any] = {"action_id": {"const": action.name}}
        required = ["action_id"]
        if action.name == "ACTION6":
            properties["target"] = {
                "type": "string",
                "description": (
                    "Concise visual description of the object or area targeted by "
                    "ACTION6."
                ),
            }
            properties["bbox"] = {
                "type": "array",
                "description": (
                    "Target bounding box in crop-relative normalized coordinates "
                    "[x0, y0, x1, y1] from 0 to 1000."
                ),
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            }
            properties["target_rgb_color"] = {
                "type": "array",
                "description": "Target RGB color as [r, g, b] integers.",
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                "minItems": 3,
                "maxItems": 3,
            }
            required.extend(["target", "bbox", "target_rgb_color"])
        branches.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    if not branches:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}


def agent_game_updated_context_json_schema(
    *,
    allowed_actions: Sequence[ActionSpec],
    actions_window: int = 1,
) -> dict[str, Any]:
    """Return the agent game updater output JSON schema."""

    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    return {
        "type": "object",
        "properties": {
            "current_strategy": {"type": "string"},
            "next_actions": {
                "type": "array",
                "items": _action_output_schema(allowed_actions),
                "minItems": actions_window,
                "maxItems": actions_window,
            },
        },
        "required": [*AGENT_GAME_CONTEXT_KEYS, "next_actions"],
        "additionalProperties": False,
    }


def updater_output_json_schema(
    task: UpdaterTask,
    *,
    allowed_actions: Sequence[ActionSpec] = (),
    actions_window: int = 1,
) -> dict[str, Any]:
    """Return the provider-neutral output schema for one updater task."""

    if task == "agent":
        return agent_game_updated_context_json_schema(
            allowed_actions=allowed_actions,
            actions_window=actions_window,
        )
    return updated_context_json_schema()


@dataclass(slots=True)
class UpdaterContextTarget:
    """Specific role context segment selected by orchestration."""

    role: UpdaterRole
    segment: ContextSegment
    task: UpdaterTask
    previous_context: RoleContext


@dataclass(slots=True)
class PromptImage:
    """Provider-neutral image attached to a prompt updater request."""

    label: str
    image: Any


@dataclass(slots=True)
class PromptUpdateRequest:
    """Provider-neutral prompt update request built for a concrete target."""

    target: UpdaterContextTarget
    instructions: str
    text: str
    output_schema: dict[str, Any]
    images: tuple[PromptImage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptUpdateResult:
    """Provider-neutral prompt update result from one updater backend."""

    target: UpdaterContextTarget
    updated_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptUpdateProviderResponse:
    """Raw provider output for one prompt update request."""

    target: UpdaterContextTarget
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentGameContextUpdateInput:
    """Input for updating the agent game-specific context document."""

    current_observation: Observation
    allowed_actions: tuple[ActionSpec, ...]
    glossary_actions: tuple[ActionSpec, ...]
    previous_context: RoleContext = field(default_factory=RoleContext)
    world_model_context: str = ""
    previous_actions_summary: str = ""
    previous_strategy_summary: str = ""
    action_history: tuple[ActionHistoryItem, ...] = ()
    previous_game_context_history: tuple[str, ...] = ()
    reset_notice: str = ""
    actions_window: int = 1


@dataclass(slots=True)
class AgentGameContextUpdateResult:
    """Selected agent updater result plus the action to queue."""

    context: str
    next_actions: tuple[ActionSpec, ...]


class AgentContextUpdaterModel(Protocol):
    """Updater task for agent strategy context and next actions."""

    def update_agent_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Return the next strategy state and selected actions."""
        ...


@dataclass(slots=True)
class UpdaterTaskRegistry:
    """Configured updater task instances for runtime orchestration."""

    agent_updater: AgentContextUpdaterModel | None = None

    def require_agent_updater(self) -> AgentContextUpdaterModel:
        """Return the agent updater, failing if not wired."""

        if self.agent_updater is None:
            raise RuntimeError("agent updater is not registered")
        return self.agent_updater
