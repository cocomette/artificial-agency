"""Provider-neutral model role boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.change.contracts import ChangeSummaryModel
from face_of_agi.models.historizer.contracts import AgentContextHistorizerModel
from face_of_agi.models.level_summary.contracts import LevelSolutionSummarizerModel
from face_of_agi.models.orchestrator_agent.contracts import OrchestratorAgentModel
from face_of_agi.models.updater.contracts import UpdaterTaskRegistry
from face_of_agi.models.world.contracts import AgentWorldModel


@dataclass(slots=True)
class ModelRegistry:
    """Small registry for injected model role implementations."""

    agent_context_historizer_model: AgentContextHistorizerModel | None = None
    level_solution_summarizer: LevelSolutionSummarizerModel | None = None
    world_model: AgentWorldModel | None = None
    orchestrator_agent: OrchestratorAgentModel | None = None
    change_summary_model: ChangeSummaryModel | None = None
    updater_tasks: UpdaterTaskRegistry | None = None

    def require_agent_context_historizer_model(self) -> AgentContextHistorizerModel:
        """Return the agent context historizer, failing early if not wired."""

        if self.agent_context_historizer_model is None:
            raise RuntimeError("agent context historizer model is not registered")
        return self.agent_context_historizer_model

    def require_world_model(self) -> AgentWorldModel:
        """Return the world-model role, failing early if not wired."""

        if self.world_model is None:
            raise RuntimeError("world model is not registered")
        return self.world_model

    def require_level_solution_summarizer(self) -> LevelSolutionSummarizerModel:
        """Return the level summarizer, failing early if it was not wired."""

        if self.level_solution_summarizer is None:
            raise RuntimeError("level solution summarizer model is not registered")
        return self.level_solution_summarizer

    def require_orchestrator_agent(self) -> OrchestratorAgentModel:
        """Return the X agent role, failing early if it was not wired."""

        if self.orchestrator_agent is None:
            raise RuntimeError("orchestrator agent model is not registered")
        return self.orchestrator_agent

    def require_change_summary_model(self) -> ChangeSummaryModel:
        """Return the change summary role, failing early if it was not wired."""

        if self.change_summary_model is None:
            raise RuntimeError("change summary model is not registered")
        return self.change_summary_model

    def require_updater_tasks(self) -> UpdaterTaskRegistry:
        """Return updater task registry, failing early if it was not wired."""

        if self.updater_tasks is None:
            raise RuntimeError("updater task registry is not registered")
        return self.updater_tasks


__all__ = [
    "ChangeSummaryModel",
    "AgentContextHistorizerModel",
    "AgentWorldModel",
    "LevelSolutionSummarizerModel",
    "ModelRegistry",
    "OrchestratorAgentModel",
    "UpdaterTaskRegistry",
]
