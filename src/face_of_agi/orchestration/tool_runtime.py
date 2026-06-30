"""Controlled tool runtime exposed to Agent X during one frame turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from face_of_agi.contracts import (
    ExperimentToolInvocationResult,
    FrameTurnContext,
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
    ) -> ExperimentToolInvocationResult:
        """Execute one orchestration-owned experiment tool call."""
        ...


@dataclass(slots=True)
class OrchestrationAgentToolRuntime:
    """Small per-turn tool interface handed to Agent X.

    The object is the narrow boundary where orchestration attaches configured
    tools and stores their results in experimental memory.
    """

    run_id: str
    game_id: str
    turn_id: int
    frame_context: FrameTurnContext
    invoke_tool: ExperimentToolInvoker | None = None
    available_tool_names: tuple[ToolName, ...] = ()
    tools_enabled: bool = True

    @property
    def current_source_state_id(self) -> int | None:
        """Return the callable frame ref for the current frame source."""

        return self.frame_context.current_source_state_id

    def available_tools(self) -> tuple[ToolName, ...]:
        """Return tools X may call during this frame turn."""

        if not self.tools_enabled:
            return ()
        return self.available_tool_names

    def invoke(
        self,
        call: ToolCall,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentToolInvocationResult:
        """Execute one configured future tool through orchestration."""

        if not self.tools_enabled:
            raise RuntimeError("tool calls are disabled for this frame")
        if self.invoke_tool is None:
            raise RuntimeError("no Agent X tools are configured")
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
        )
