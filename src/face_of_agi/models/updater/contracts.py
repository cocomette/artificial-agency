"""Contracts for the updater model P."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ObservationRef,
    PostDecisionPredictions,
    RewardUpdateQuantities,
    RoleContext,
    ToolResult,
)

UpdaterToolRole = Literal["world", "goal"]


@dataclass(slots=True)
class ToolContextUpdateInput:
    """Input for updating a world or goal game-specific context document."""

    role: UpdaterToolRole
    previous_context: RoleContext
    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
    )
    tool_results: tuple[ToolResult, ...] = ()
    quantities: RewardUpdateQuantities = field(default_factory=RewardUpdateQuantities)
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContextUpdateInput:
    """Input for updating the orchestrator agent game-specific context document."""

    previous_context: RoleContext
    current_observation_ref: ObservationRef
    actual_next_observation_ref: ObservationRef | None
    trace: AgentTrace
    post_decision_predictions: PostDecisionPredictions = field(
        default_factory=PostDecisionPredictions
    )
    quantities: RewardUpdateQuantities = field(default_factory=RewardUpdateQuantities)
    submitted_action: ActionSpec | None = None
    synthetic_none_action: ActionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UpdaterModel(Protocol):
    """Updater role P for online context revisions."""

    def update_tool_context(
        self,
        update_input: ToolContextUpdateInput,
    ) -> RoleContext:
        """Return the next world or goal context document."""
        ...

    def update_agent_context(
        self,
        update_input: AgentContextUpdateInput,
    ) -> RoleContext:
        """Return the next agent context document."""
        ...
