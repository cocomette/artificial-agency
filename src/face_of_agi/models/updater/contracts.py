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
    SamePastStateDetection,
)
from face_of_agi.models.historizer import AgentContextHistorySummary

UpdaterRole = Literal["agent"]
ContextSegment = Literal["general", "game"]
AgentUpdaterMode = Literal["probing", "policy"]
UpdaterTask = Literal["agent_probing", "agent_policy", "general"]
AGENT_GAME_CONTEXT_KEYS = (
    "probing_strategy",
    "policy_strategy",
)
AGENT_GAME_OUTPUT_KEYS = {
    "probing": ("probing_strategy",),
    "policy": ("policy_strategy",),
}
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


def agent_game_output_keys(mode: AgentUpdaterMode) -> tuple[str, ...]:
    return AGENT_GAME_OUTPUT_KEYS[mode]


def agent_game_updated_context_json_schema(
    *,
    mode: AgentUpdaterMode,
    allowed_actions: Sequence[ActionSpec],
    actions_window: int = 1,
) -> dict[str, Any]:
    """Return the agent game updater output JSON schema."""

    if actions_window < 1:
        raise ValueError("actions_window must be at least 1")
    output_keys = agent_game_output_keys(mode)
    properties: dict[str, dict[str, Any]] = {}
    if mode == "probing":
        properties["probing_strategy"] = {
            "type": "string",
        }
    else:
        properties["policy_strategy"] = {
            "type": "string",
        }
    return {
        "type": "object",
        "properties": {
            **properties,
            "next_actions": {
                "type": "array",
                "items": _action_output_schema(allowed_actions),
                "minItems": actions_window,
                "maxItems": actions_window,
            },
        },
        "required": [*output_keys, "next_actions"],
        "additionalProperties": False,
    }


def updater_output_json_schema(
    task: UpdaterTask,
    *,
    allowed_actions: Sequence[ActionSpec] = (),
    actions_window: int = 1,
) -> dict[str, Any]:
    """Return the provider-neutral output schema for one updater task."""

    if task == "agent_probing":
        return agent_game_updated_context_json_schema(
            mode="probing",
            allowed_actions=allowed_actions,
            actions_window=actions_window,
        )
    if task == "agent_policy":
        return agent_game_updated_context_json_schema(
            mode="policy",
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

    previous_context: RoleContext
    current_observation: Observation
    allowed_actions: tuple[ActionSpec, ...]
    glossary_actions: tuple[ActionSpec, ...]
    context_history: AgentContextHistorySummary = field(
        default_factory=AgentContextHistorySummary.not_available
    )
    same_past_state_detections: tuple[SamePastStateDetection, ...] = ()
    previous_level_solution_method: str = ""
    action_history: tuple[ActionHistoryItem, ...] = ()
    actions_window: int = 1


@dataclass(slots=True)
class AgentGameContextUpdateResult:
    """Selected agent updater result plus the action to queue."""

    context: str
    next_actions: tuple[ActionSpec, ...]
    updater_mode: AgentUpdaterMode


@dataclass(slots=True)
class GeneralKnowledgeUpdateInput:
    """Run-level input for updating one role's game-agnostic context."""

    role: UpdaterRole
    previous_context: RoleContext = field(default_factory=RoleContext)
    run_id: str = ""
    game_id: str = ""
    stop_reason: str | None = None
    step_count: int = 0
    completed_levels: int = 0
    final_state: str | None = None
    state_record_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentProbingContextUpdaterModel(Protocol):
    """Updater task for choosing the next probing action."""

    def update_agent_probing_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Return the next probing strategy and selected action."""
        ...


class AgentPolicyContextUpdaterModel(Protocol):
    """Updater task for agent policy-strategy context."""

    def update_agent_policy_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        """Return the next policy strategy and selected action."""
        ...


class GeneralKnowledgeUpdaterModel(Protocol):
    """Shared updater task for role-specific general knowledge `K`."""

    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        """Return the next role context after an end-of-run general update."""
        ...


@dataclass(slots=True)
class UpdaterTaskRegistry:
    """Configured updater task instances for runtime orchestration."""

    agent_probing_updater: AgentProbingContextUpdaterModel | None = None
    agent_policy_updater: AgentPolicyContextUpdaterModel | None = None
    general_updater: GeneralKnowledgeUpdaterModel | None = None

    def require_agent_probing_updater(self) -> AgentProbingContextUpdaterModel:
        """Return the probing updater, failing if not wired."""

        if self.agent_probing_updater is None:
            raise RuntimeError("agent probing updater is not registered")
        return self.agent_probing_updater

    def require_agent_policy_updater(self) -> AgentPolicyContextUpdaterModel:
        """Return the policy updater, failing if not wired."""

        if self.agent_policy_updater is None:
            raise RuntimeError("agent policy updater is not registered")
        return self.agent_policy_updater

    def require_general_updater(self) -> GeneralKnowledgeUpdaterModel:
        """Return the shared general updater, failing if not wired."""

        if self.general_updater is None:
            raise RuntimeError("general updater is not registered")
        return self.general_updater
