"""Contracts for the updater model P."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    TurnMetrics,
    RoleContext,
    ToolResult,
)

UpdaterRole = Literal["world", "goal", "agent"]
ContextSegment = Literal["general", "game"]
UpdaterTask = Literal["world_game", "goal_game", "agent_game", "general"]


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
    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    previous_observation: Observation | None = None
    actual_next_observation: Observation | None = None
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
    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    previous_observation: Observation | None = None
    actual_next_observation: Observation | None = None
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
    """Progress feedback visible to the agent game-context updater."""

    time_cost: float | None = None
    score_delta: float | None = None


@dataclass(slots=True)
class AgentGameContextUpdateInput:
    """Input for updating the agent game-specific context document."""

    previous_context: RoleContext
    previous_observation: Observation
    current_observation: Observation
    current_turn_world_game_context: str
    current_turn_goal_game_context: str
    previous_turn_world_game_context: str | None
    trace: AgentTrace
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
