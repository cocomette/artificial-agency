"""Orchestration-owned router for world and goal prediction calls."""

from __future__ import annotations

from face_of_agi.contracts import (
    Observation,
    ObservationRef,
    PredictionCall,
    PredictionResult,
    RoleContext,
)
from face_of_agi.models.goal.contracts import GoalPredictionModel
from face_of_agi.models.world.contracts import WorldPredictionModel


class PredictionRouter:
    """Dispatch orchestration-owned calls to world and goal model roles."""

    def __init__(
        self,
        *,
        world_model: WorldPredictionModel | None = None,
        goal_model: GoalPredictionModel | None = None,
    ) -> None:
        self.world_model = world_model
        self.goal_model = goal_model

    def route(
        self,
        *,
        call: PredictionCall,
        context: RoleContext,
        observation: Observation,
    ) -> PredictionResult:
        """Route a single orchestration-owned prediction call."""

        if call.tool == "world":
            if self.world_model is None:
                raise RuntimeError("world model is not registered")
            if call.action is None:
                raise ValueError("world model prediction calls require an action")
            result = self.world_model.predict(context, call.action, observation)
            result.source_observation_ref = ObservationRef(
                memory="state",
                id=observation.id,
            )
            result.source_state_id = call.source_state_id
            return result

        if self.goal_model is None:
            raise RuntimeError("goal model is not registered")
        result = self.goal_model.predict(context, observation)
        result.source_observation_ref = ObservationRef(memory="state", id=observation.id)
        result.source_state_id = call.source_state_id
        return result
