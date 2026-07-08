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
    """Produce S predictions after X chooses a frame action."""

    def __init__(
        self,
        *,
        world_model: WorldPredictionModel | None = None,
        debug: DebugBus | None = None,
    ) -> None:
        self.world_model = world_model
        self.debug = debug or DebugBus.disabled()

    def predict(
        self,
        *,
        current_observation_ref: ObservationRef,
        current_source_state_id: int | None,
        current_observation: Observation,
        final_action: ActionSpec,
        world_context: RoleContext,
    ) -> PostDecisionPredictions:
        """Return world predictions for a committed frame decision."""

        if self.world_model is None:
            raise RuntimeError("world model is not registered")

        router = PredictionRouter(
            world_model=self.world_model,
        )
        world_call = PredictionCall(
            tool="world",
            source_state_id=current_source_state_id or 0,
            action=final_action,
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
        world_prediction.metadata = {
            **world_prediction.metadata,
            "purpose": "post_decision_update_prediction",
        }
        self.debug.emit(
            ToolResultRecorded(
                role="world",
                purpose="post_decision_update_prediction",
                result=world_prediction,
            )
        )
        return PostDecisionPredictions(
            world_prediction=world_prediction,
        )
