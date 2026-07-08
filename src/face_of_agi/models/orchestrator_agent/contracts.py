"""Contracts for the orchestrator agent model X."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from face_of_agi.contracts import (
    ActionSpec,
    DecisionResult,
    ExperimentToolInvocationResult,
    Observation,
    ObservationRef,
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
    def first_observation_ref(self) -> ObservationRef:
        """Return the first real observation reference visible to X."""
        ...

    @property
    def current_observation_ref(self) -> ObservationRef:
        """Return the current real observation reference visible to X."""
        ...

    def available_observation_refs(self) -> tuple[ObservationRef, ...]:
        """Return memory refs that are immediately visible to X."""
        ...

    def available_tools(self) -> tuple[ToolName, ...]:
        """Return model tools available for this frame turn."""
        ...

    def tool_metadata(self) -> dict[str, Any]:
        """Return frame-local tool policy metadata."""
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
    """Agent role X that may call world and goal model tools."""

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
    ) -> DecisionResult:
        """Return one final action and its decision trace."""
        ...
