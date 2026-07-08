"""Random shell provider for orchestrator agent X."""

from __future__ import annotations

import random
from collections.abc import Sequence

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    Observation,
    ObservationRef,
    RoleContext,
)
from face_of_agi.models.orchestrator_agent.config import OrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.contracts import AgentToolRuntime


class RandomOrchestratorAgentAdapter:
    """Empty/random X shell used when no reasoning backend is configured."""

    def __init__(
        self,
        config: OrchestratorAgentConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or OrchestratorAgentConfig()
        self.rng = rng or random.Random()

    def decide(
        self,
        context: RoleContext,
        first_observation: Observation,
        current_observation: Observation,
        action_space: Sequence[ActionSpec],
        tool_runtime: AgentToolRuntime | None = None,
    ) -> DecisionResult:
        """Select one final action from the framework-provided action space."""

        del context
        if not action_space:
            raise RuntimeError("orchestrator agent received no valid actions")

        chosen_action = self.rng.choice(tuple(action_space))
        if chosen_action.is_complex():
            chosen_action = ActionSpec(
                action_id=chosen_action.action_id,
                data={
                    "x": self.rng.randint(0, 63),
                    "y": self.rng.randint(0, 63),
                },
            )

        first_ref = ObservationRef(memory="state", id=first_observation.id)
        current_ref = ObservationRef(memory="state", id=current_observation.id)
        trace = AgentTrace(
            step=current_observation.step,
            first_observation_ref=first_ref,
            current_observation_ref=current_ref,
            final_action=chosen_action,
            reasoning_summary="empty X shell selected from provided action space",
            metadata={
                "model_role": "X",
                "shell": "random",
                "tool_runtime_available": tool_runtime is not None,
            },
        )
        return DecisionResult(final_action=chosen_action, trace=trace)
