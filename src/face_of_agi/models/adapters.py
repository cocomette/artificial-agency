"""Provider-neutral model role boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.goal.contracts import GoalPredictionModel
from face_of_agi.models.orchestrator_agent.contracts import OrchestratorAgentModel
from face_of_agi.models.updater.contracts import UpdaterTaskRegistry
from face_of_agi.models.world.contracts import WorldPredictionModel


@dataclass(slots=True)
class ModelRegistry:
    """Small registry for injected model role implementations."""

    world_prediction_model: WorldPredictionModel | None = None
    goal_prediction_model: GoalPredictionModel | None = None
    orchestrator_agent: OrchestratorAgentModel | None = None
    updater_tasks: UpdaterTaskRegistry | None = None

    def require_world_prediction_model(self) -> WorldPredictionModel:
        """Return the world prediction role, failing early if it was not wired."""

        if self.world_prediction_model is None:
            raise RuntimeError("world prediction model is not registered")
        return self.world_prediction_model

    def require_goal_prediction_model(self) -> GoalPredictionModel:
        """Return the goal prediction role, failing early if it was not wired."""

        if self.goal_prediction_model is None:
            raise RuntimeError("goal prediction model is not registered")
        return self.goal_prediction_model

    def require_orchestrator_agent(self) -> OrchestratorAgentModel:
        """Return the X agent role, failing early if it was not wired."""

        if self.orchestrator_agent is None:
            raise RuntimeError("orchestrator agent model is not registered")
        return self.orchestrator_agent

    def require_updater_tasks(self) -> UpdaterTaskRegistry:
        """Return updater task registry, failing early if it was not wired."""

        if self.updater_tasks is None:
            raise RuntimeError("updater task registry is not registered")
        return self.updater_tasks


__all__ = [
    "GoalPredictionModel",
    "ModelRegistry",
    "OrchestratorAgentModel",
    "UpdaterTaskRegistry",
    "WorldPredictionModel",
]
