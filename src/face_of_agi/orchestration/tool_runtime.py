"""Controlled tool runtime exposed to Agent X during one frame turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from face_of_agi.contracts import (
    ExperimentToolInvocationResult,
    FrameTurnContext,
    Observation,
    ObservationRef,
    ToolCall,
    ToolName,
)


class ExperimentToolInvoker(Protocol):
    """Callable shape used by the per-turn agent tool runtime."""

    def __call__(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        frame_context: FrameTurnContext,
        call: ToolCall,
        metadata: dict[str, Any] | None = None,
        state_observations: Mapping[str, Observation] | None = None,
    ) -> ExperimentToolInvocationResult:
        """Execute one orchestration-owned experiment tool call."""
        ...


@dataclass(slots=True)
class OrchestrationAgentToolRuntime:
    """Small per-turn tool interface handed to Agent X.

    Agent X can request tool calls through this object, but orchestration still
    owns reference resolution, model routing, E writes, and rolling cleanup.
    """

    run_id: str
    game_id: str
    turn_id: int
    frame_context: FrameTurnContext
    invoke_tool: ExperimentToolInvoker
    state_observations: Mapping[str, Observation]
    available_tool_names: tuple[ToolName, ...] = ()
    tools_enabled: bool = True

    @property
    def first_observation_ref(self) -> ObservationRef:
        """Return the first real observation reference visible to X."""

        return self.frame_context.first_observation_ref

    @property
    def current_observation_ref(self) -> ObservationRef:
        """Return the current real observation reference visible to X."""

        return self.frame_context.current_observation_ref

    def available_observation_refs(self) -> tuple[ObservationRef, ...]:
        """Return real observation refs immediately visible to X this turn."""

        refs = (
            self.frame_context.first_observation_ref,
            self.frame_context.current_observation_ref,
        )
        deduped: list[ObservationRef] = []
        for ref in refs:
            if ref not in deduped:
                deduped.append(ref)
        return tuple(deduped)

    def available_tools(self) -> tuple[ToolName, ...]:
        """Return model tools X may call during this frame turn."""

        if not self.tools_enabled:
            return ()
        return self.available_tool_names

    def tool_metadata(self) -> dict[str, Any]:
        """Return frame-local tool policy details for agent prompts."""

        return {
            "tools_enabled": self.tools_enabled,
            "available_tools": list(self.available_tools()),
            "control_mode": {
                "controllable": self.frame_context.control_mode.controllable,
                "reason": self.frame_context.control_mode.reason,
            },
        }

    def invoke(
        self,
        call: ToolCall,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentToolInvocationResult:
        """Execute one tool call through orchestration and persist it in E."""

        if not self.tools_enabled:
            raise RuntimeError("tool calls are disabled for this frame")
        if call.tool not in self.available_tool_names:
            raise RuntimeError(f"{call.tool} tool is not available for this frame")

        merged_metadata = {"requested_by": "agent_x", **(metadata or {})}
        return self.invoke_tool(
            run_id=self.run_id,
            game_id=self.game_id,
            turn_id=self.turn_id,
            frame_context=self.frame_context,
            call=call,
            metadata=merged_metadata,
            state_observations=self.state_observations,
        )
