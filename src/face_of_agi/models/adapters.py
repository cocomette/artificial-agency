"""Provider-neutral model role boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from face_of_agi.models.change.contracts import ChangeSummaryModel
from face_of_agi.models.goal.contracts import GoalModel
from face_of_agi.models.historizer.contracts import AgentContextHistorizerModel
from face_of_agi.models.interest.contracts import InterestModel
from face_of_agi.models.memory.contracts import MemoryModel
from face_of_agi.models.orchestrator_agent.contracts import OrchestratorAgentModel
from face_of_agi.models.reward_judge.contracts import RewardJudgeModel
from face_of_agi.models.updater.contracts import UpdaterTaskRegistry
from face_of_agi.models.world.contracts import WorldModel


@dataclass(slots=True)
class ModelRegistry:
    """Small registry for injected model role implementations."""

    agent_context_historizer_model: AgentContextHistorizerModel | None = None
    orchestrator_agent: OrchestratorAgentModel | None = None
    change_summary_model: ChangeSummaryModel | None = None
    updater_tasks: UpdaterTaskRegistry | None = None
    memory_model: MemoryModel | None = None
    world_model: WorldModel | None = None
    goal_model: GoalModel | None = None
    interest_model: InterestModel | None = None
    reward_judge_model: RewardJudgeModel | None = None

    def require_agent_context_historizer_model(self) -> AgentContextHistorizerModel:
        """Return the agent context historizer, failing early if not wired."""

        if self.agent_context_historizer_model is None:
            raise RuntimeError("agent context historizer model is not registered")
        return self.agent_context_historizer_model

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

    def require_memory_model(self) -> MemoryModel:
        """Return the Memory role, failing early if it was not wired."""

        if self.memory_model is None:
            raise RuntimeError("memory model is not registered")
        return self.memory_model

    def require_world_model(self) -> WorldModel:
        """Return the World role, failing early if it was not wired."""

        if self.world_model is None:
            raise RuntimeError("world model is not registered")
        return self.world_model

    def require_goal_model(self) -> GoalModel:
        """Return the Goal role, failing early if it was not wired."""

        if self.goal_model is None:
            raise RuntimeError("goal model is not registered")
        return self.goal_model

    def require_interest_model(self) -> InterestModel:
        """Return the Interest role, failing early if it was not wired."""

        if self.interest_model is None:
            raise RuntimeError("interest model is not registered")
        return self.interest_model

    def require_reward_judge_model(self) -> RewardJudgeModel:
        """Return the Reward Judge role, failing early if it was not wired."""

        if self.reward_judge_model is None:
            raise RuntimeError("reward judge model is not registered")
        return self.reward_judge_model


__all__ = [
    "ChangeSummaryModel",
    "GoalModel",
    "AgentContextHistorizerModel",
    "InterestModel",
    "MemoryModel",
    "ModelRegistry",
    "OrchestratorAgentModel",
    "RewardJudgeModel",
    "UpdaterTaskRegistry",
    "WorldModel",
]
