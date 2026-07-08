"""Contracts for the orchestrator agent model X."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    RoleContext,
    ToolCall,
    ToolName,
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

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
        world_game_context: str = "",
        goal_game_context: str = "",
        recent_action_history: tuple[ActionHistoryEntry, ...] = (),
    ) -> DecisionResult:
        """Return one final action and its decision trace."""
        ...
