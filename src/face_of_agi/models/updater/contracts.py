"""Contracts for the updater model P."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    PostDecisionPredictions,
    TurnMetrics,
    RoleContext,
    ToolResult,
)

UpdaterRole = Literal["world", "goal", "agent"]
ContextSegment = Literal["general", "game"]
UpdaterTask = Literal["world_game", "goal_game", "agent_game", "general"]
WORLD_GAME_ACTION_KEYS = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "NONE",
)
WORLD_GAME_CONTEXT_KEYS = ("world_understanding", *WORLD_GAME_ACTION_KEYS)
WORLD_GAME_UNDERSTANDING_DESCRIPTION = (
    "Updated general game environment world understanding."
)
WORLD_GAME_ACTION_DESCRIPTIONS = (
    "Initialize or restart the game or level state.",
    "Arrow up.",
    "Arrow down.",
    "Arrow left.",
    "Arrow right.",
    "Simple game-specific action: interact, select, rotate, attach/detach, or execute.",
    "Coordinate action targeting x,y.",
    "Undo-style simple action.",
    "Internal no-control action for animation-frame unrolling.",
)
WORLD_GAME_ACTION_DESCRIPTION_BY_KEY = dict(
    zip(WORLD_GAME_ACTION_KEYS, WORLD_GAME_ACTION_DESCRIPTIONS, strict=True)
)
AGENT_GAME_CONTEXT_KEYS = (
    "goals",
    "game_mechanics",
    "policy",
    "history",
    "extras",
)


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


def world_game_updated_context_json_schema() -> dict[str, Any]:
    """Return the world game updater output JSON schema."""

    return {
        "type": "object",
        "properties": {
            "updated_context": {
                "type": "object",
                "description": (
                    "Complete latest action-effect context, keyed by every "
                    "action in the world updater action glossary."
                ),
                "properties": {
                    key: {
                        "type": "string",
                        "description": (
                            WORLD_GAME_UNDERSTANDING_DESCRIPTION
                            if key == "world_understanding"
                            else WORLD_GAME_ACTION_DESCRIPTION_BY_KEY[key]
                        ),
                    }
                    for key in WORLD_GAME_CONTEXT_KEYS
                },
                "required": list(WORLD_GAME_CONTEXT_KEYS),
                "additionalProperties": False,
            },
        },
        "required": ["updated_context"],
        "additionalProperties": False,
    }


def agent_game_updated_context_json_schema() -> dict[str, Any]:
    """Return the agent game updater output JSON schema."""

    descriptions = {
        "goals": "Current objective, progress target, and goal hypothesis.",
        "game_mechanics": "Useful world/action dynamics and uncertainty.",
        "policy": "General guidance for how to approach playing this game.",
        "history": "Compact learnings from past outcomes and progress evidence.",
        "extras": "Any other useful agent guidance.",
    }
    return {
        "type": "object",
        "properties": {
            "updated_context": {
                "type": "object",
                "description": "Complete latest agent game context.",
                "properties": {
                    key: {
                        "type": "string",
                        "description": descriptions[key],
                    }
                    for key in AGENT_GAME_CONTEXT_KEYS
                },
                "required": list(AGENT_GAME_CONTEXT_KEYS),
                "additionalProperties": False,
            },
        },
        "required": ["updated_context"],
        "additionalProperties": False,
    }


def updater_output_json_schema(task: UpdaterTask) -> dict[str, Any]:
    """Return the provider-neutral output schema for one updater task."""

    if task == "world_game":
        return world_game_updated_context_json_schema()
    if task == "agent_game":
        return agent_game_updated_context_json_schema()
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
class WorldGameContextUpdateInput:
    """Input for updating the world model game-specific context document."""

    previous_context: RoleContext
    current_observation: Observation
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
    )
    tool_results: tuple[ToolResult, ...] = ()
    turn_metrics: TurnMetrics = field(default_factory=TurnMetrics)
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoalGameContextUpdateInput:
    """Input for updating the goal model game-specific context document."""

    previous_context: RoleContext
    current_observation: Observation
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
    )
    tool_results: tuple[ToolResult, ...] = ()
    turn_metrics: TurnMetrics = field(default_factory=TurnMetrics)
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentProgressFeedback:
    """Progress and compactness feedback visible to the agent updater."""

    time_cost: float | None = None
    cumulative_score: float | None = None
    agent_context_word_count: int | None = None


@dataclass(slots=True)
class AgentGameContextUpdateInput:
    """Input for updating the agent game-specific context document."""

    previous_context: RoleContext
    previous_observation: Observation
    current_observation: Observation
    current_turn_world_game_context: str
    previous_turn_world_game_context: str | None
    action_history: tuple[ActionHistoryEntry, ...] = ()
    turn_metrics: AgentProgressFeedback = field(
        default_factory=AgentProgressFeedback
    )


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


class WorldGameContextUpdaterModel(Protocol):
    """Updater task for world game context `L^S`."""

    def update_world_game_context(
        self,
        update_input: WorldGameContextUpdateInput,
    ) -> RoleContext:
        """Return the next world game context document."""
        ...


class GoalGameContextUpdaterModel(Protocol):
    """Updater task for goal game context `L^G`."""

    def update_goal_game_context(
        self,
        update_input: GoalGameContextUpdateInput,
    ) -> RoleContext:
        """Return the next goal game context document."""
        ...


class AgentGameContextUpdaterModel(Protocol):
    """Updater task for agent game context `L^X`."""

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        """Return the next agent game context document."""
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

    world_game_updater: WorldGameContextUpdaterModel | None = None
    goal_game_updater: GoalGameContextUpdaterModel | None = None
    agent_game_updater: AgentGameContextUpdaterModel | None = None
    general_updater: GeneralKnowledgeUpdaterModel | None = None

    def require_world_game_updater(self) -> WorldGameContextUpdaterModel:
        """Return the world game updater, failing if not wired."""

        if self.world_game_updater is None:
            raise RuntimeError("world game updater is not registered")
        return self.world_game_updater

    def require_goal_game_updater(self) -> GoalGameContextUpdaterModel:
        """Return the goal game updater, failing if not wired."""

        if self.goal_game_updater is None:
            raise RuntimeError("goal game updater is not registered")
        return self.goal_game_updater

    def require_agent_game_updater(self) -> AgentGameContextUpdaterModel:
        """Return the agent game updater, failing if not wired."""

        if self.agent_game_updater is None:
            raise RuntimeError("agent game updater is not registered")
        return self.agent_game_updater

    def require_general_updater(self) -> GeneralKnowledgeUpdaterModel:
        """Return the shared general updater, failing if not wired."""

        if self.general_updater is None:
            raise RuntimeError("general updater is not registered")
        return self.general_updater
