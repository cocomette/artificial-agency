"""Post-decision prediction runner owned by orchestration."""

from __future__ import annotations

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    PredictionCall,
    PostDecisionPredictions,
    RoleContext,
)
from face_of_agi.models.goal.contracts import GoalPredictionModel
from face_of_agi.models.world.contracts import WorldPredictionModel
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    ToolModelInputCaptured,
    ToolProviderInputCaptured,
    ToolResultRecorded,
)
from face_of_agi.orchestration.prediction_router import PredictionRouter
from face_of_agi.runtime import timing as runtime_timing


class PostDecisionPredictionRunner:
    """Produce S/G predictions after X chooses a frame action."""

    def __init__(
        self,
        *,
        world_model: WorldPredictionModel | None = None,
        goal_model: GoalPredictionModel | None = None,
        debug: DebugBus | None = None,
    ) -> None:
        self.world_model = world_model
        self.goal_model = goal_model
        self.debug = debug or DebugBus.disabled()

    def predict(
        self,
        *,
        current_observation_ref: ObservationRef,
        current_source_state_id: int | None,
        current_observation: Observation,
        final_action: ActionSpec,
        world_context: RoleContext,
        goal_context: RoleContext,
    ) -> PostDecisionPredictions:
        """Return world and goal predictions for a committed frame decision."""

        if self.world_model is None:
            raise RuntimeError("world model is not registered")
        if self.goal_model is None:
            raise RuntimeError("goal model is not registered")

        router = PredictionRouter(
            world_model=self.world_model,
            goal_model=self.goal_model,
        )
        world_call = PredictionCall(
            tool="world",
            source_state_id=current_source_state_id or 0,
            action=final_action,
        )
        goal_call = PredictionCall(
            tool="goal",
            source_state_id=current_source_state_id or 0,
        )
        self.debug.emit(
            ToolModelInputCaptured(
                role="world",
                purpose="post_decision_update_prediction",
                call=world_call,
                context=world_context,
                observation=current_observation,
            )
        )
        with runtime_timing.span("post_decision.world_prediction"):
            world_prediction = router.route(
                call=world_call,
                context=world_context,
                observation=current_observation,
            )
        self.debug.emit(
            ToolProviderInputCaptured(
                role="world",
                purpose="post_decision_update_prediction",
                adapter=self.world_model,
            )
        )
        self.debug.emit(
            ToolModelInputCaptured(
                role="goal",
                purpose="post_decision_update_prediction",
                call=goal_call,
                context=goal_context,
                observation=current_observation,
            )
        )
        with runtime_timing.span("post_decision.goal_prediction"):
            goal_prediction = router.route(
                call=goal_call,
                context=goal_context,
                observation=current_observation,
            )
        self.debug.emit(
            ToolProviderInputCaptured(
                role="goal",
                purpose="post_decision_update_prediction",
                adapter=self.goal_model,
            )
        )
        world_prediction.metadata = {
            **world_prediction.metadata,
            "purpose": "post_decision_update_prediction",
        }
        goal_prediction.metadata = {
            **goal_prediction.metadata,
            "purpose": "post_decision_update_prediction",
        }
        self.debug.emit(
            ToolResultRecorded(
                role="world",
                purpose="post_decision_update_prediction",
                result=world_prediction,
            )
        )
        self.debug.emit(
            ToolResultRecorded(
                role="goal",
                purpose="post_decision_update_prediction",
                result=goal_prediction,
            )
        )
        return PostDecisionPredictions(
            world_prediction=world_prediction,
            goal_prediction=goal_prediction,
        )
