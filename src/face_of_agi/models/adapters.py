"""Provider-neutral model role boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.orchestrator_agent.contracts import OrchestratorAgentModel
from face_of_agi.models.tools.goal.contracts import GoalToolModel
from face_of_agi.models.tools.world.contracts import WorldToolModel
from face_of_agi.models.updater.contracts import UpdaterModel


@dataclass(slots=True)
class ModelRegistry:
    """Small registry for injected model role implementations."""

    world_tool: WorldToolModel | None = None
    goal_tool: GoalToolModel | None = None
    orchestrator_agent: OrchestratorAgentModel | None = None
    updater: UpdaterModel | None = None

    def require_world_tool(self) -> WorldToolModel:
        """Return the world model tool role, failing early if it was not wired."""

        if self.world_tool is None:
            raise RuntimeError("world model tool is not registered")
        return self.world_tool

    def require_goal_tool(self) -> GoalToolModel:
        """Return the goal model tool role, failing early if it was not wired."""

        if self.goal_tool is None:
            raise RuntimeError("goal model tool is not registered")
        return self.goal_tool

    def require_orchestrator_agent(self) -> OrchestratorAgentModel:
        """Return the X agent role, failing early if it was not wired."""

        if self.orchestrator_agent is None:
            raise RuntimeError("orchestrator agent model is not registered")
        return self.orchestrator_agent

    def require_updater(self) -> UpdaterModel:
        """Return the updater role, failing early if it was not wired."""

        if self.updater is None:
            raise RuntimeError("updater model is not registered")
        return self.updater


__all__ = [
    "GoalToolModel",
    "ModelRegistry",
    "OrchestratorAgentModel",
    "UpdaterModel",
    "WorldToolModel",
]
