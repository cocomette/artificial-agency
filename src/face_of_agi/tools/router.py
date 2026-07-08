"""Tool router boundary for future interactive agent calls."""

from __future__ import annotations

from face_of_agi.contracts import Observation, RoleContext, ToolCall, ToolResult
from face_of_agi.models.tools.goal.contracts import GoalToolModel
from face_of_agi.models.tools.world.contracts import WorldToolModel


class ToolRouter:
    """Dispatch tool calls to injected world and goal model roles."""

    def __init__(
        self,
        *,
        world_tool: WorldToolModel | None = None,
        goal_tool: GoalToolModel | None = None,
    ) -> None:
        self.world_tool = world_tool
        self.goal_tool = goal_tool

    def route(
        self,
        *,
        call: ToolCall,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        """Route a single tool call.

        The deterministic loop will decide when agent/tool interaction is
        enabled.
        """

        if call.tool == "world":
            if self.world_tool is None:
                raise RuntimeError("world model is not registered")
            if call.action is None:
                raise ValueError("world model tool calls require an action")
            return self.world_tool.predict(context, call.action, observation)

        if self.goal_tool is None:
            raise RuntimeError("goal model is not registered")
        return self.goal_tool.predict(context, observation)
