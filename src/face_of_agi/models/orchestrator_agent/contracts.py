"""Contracts for the orchestrator agent model X."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from face_of_agi.contracts import (
    AgentCandidateAction,
    ActionHistoryItem,
    ActionOutcomeEvidence,
    ActionSpec,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    ObservationRef,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    RoleContext,
    ToolCall,
    ToolName,
    WorldPrediction,
)


class AgentToolRuntime(Protocol):
    """Controlled tool boundary exposed to Agent X during one frame turn."""

    @property
    def turn_id(self) -> int:
        """Return the current orchestration frame-turn id."""
        ...

    @property
    def current_source_state_id(self) -> int | None:
        """Return the callable frame ref for the current source."""
        ...

    def available_tools(self) -> tuple[ToolName, ...]:
        """Return model tools available for this frame turn."""
        ...

    def invoke(
        self,
        call: ToolCall,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentToolInvocationResult:
        """Execute one requested tool call through orchestration."""
        ...


class OrchestratorAgentModel(Protocol):
    """Agent role X that chooses final actions."""

    def propose_candidate_actions(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        max_candidates: int,
        recent_action_history: tuple[ActionHistoryItem, ...] = (),
        glossary_actions: Sequence[ActionSpec],
    ) -> tuple[AgentCandidateAction, ...]:
        """Return distinct coordinate candidates for the v1 two-stage loop."""
        ...

    def select_action(
        self,
        *,
        memory: MemoryDocument,
        goal: GoalPrediction,
        current_observation: Observation,
        candidates: Sequence[AgentCandidateAction],
        world_predictions: Sequence[WorldPrediction],
        interest_prediction: InterestPrediction | None = None,
        first_observation_ref: ObservationRef | None = None,
        recent_action_history: tuple[ActionHistoryItem, ...] = (),
        glossary_actions: Sequence[ActionSpec],
    ) -> DecisionResult:
        """Select one final action from world-evaluated candidates."""
        ...

    def decide(
        self,
        context: RoleContext,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
        recent_action_history: tuple[ActionHistoryItem, ...] = (),
        *,
        glossary_actions: Sequence[ActionSpec],
        first_observation_ref: ObservationRef | None = None,
        recent_action_history_available: bool = True,
        action_outcome_evidence: ActionOutcomeEvidence | None = None,
    ) -> DecisionResult:
        """Return one final action and its decision trace."""
        ...
