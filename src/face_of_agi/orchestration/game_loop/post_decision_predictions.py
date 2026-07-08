"""Post-decision prediction runner owned by orchestration."""

from __future__ import annotations

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.frames import normalize_frame_for_memory
from face_of_agi.models.tools.goal.contracts import GoalToolModel
from face_of_agi.models.tools.world.contracts import WorldToolModel
from face_of_agi.tools import ToolRouter


class PostDecisionPredictionRunner:
    """Produce committed S/G predictions after X chooses a real action."""

    def __init__(
        self,
        *,
        world_tool: WorldToolModel | None = None,
        goal_tool: GoalToolModel | None = None,
        prompt_model_calls_enabled: bool = False,
    ) -> None:
        self.world_tool = world_tool
        self.goal_tool = goal_tool
        self.prompt_model_calls_enabled = prompt_model_calls_enabled

    def predict(
        self,
        *,
        current_observation_ref: ObservationRef,
        current_observation: Observation,
        final_action: ActionSpec,
        world_context: RoleContext,
        goal_context: RoleContext,
    ) -> PostDecisionPredictions:
        """Return world and goal predictions for a committed decision."""

        if not self.prompt_model_calls_enabled:
            return self._mock_predictions(
                current_observation_ref=current_observation_ref,
                current_observation=current_observation,
                final_action=final_action,
            )

        if self.world_tool is None:
            raise RuntimeError("world model is not registered")
        if self.goal_tool is None:
            raise RuntimeError("goal model is not registered")

        router = ToolRouter(
            world_tool=self.world_tool,
            goal_tool=self.goal_tool,
        )
        world_call = ToolCall(
            tool="world",
            observation_ref=current_observation_ref,
            action=final_action,
        )
        goal_call = ToolCall(
            tool="goal",
            observation_ref=current_observation_ref,
        )
        world_prediction = router.route(
            call=world_call,
            context=world_context,
            observation=current_observation,
        )
        goal_prediction = router.route(
            call=goal_call,
            context=goal_context,
            observation=current_observation,
        )
        world_prediction.predicted_observation = normalize_frame_for_memory(
            world_prediction.predicted_observation
        )
        goal_prediction.predicted_observation = normalize_frame_for_memory(
            goal_prediction.predicted_observation
        )
        world_prediction.metadata = {
            **world_prediction.metadata,
            "purpose": "post_decision_update_prediction",
            "prompt_model_calls_enabled": True,
        }
        goal_prediction.metadata = {
            **goal_prediction.metadata,
            "purpose": "post_decision_update_prediction",
            "prompt_model_calls_enabled": True,
        }
        return PostDecisionPredictions(
            world_prediction=world_prediction,
            goal_prediction=goal_prediction,
        )

    def _mock_predictions(
        self,
        *,
        current_observation_ref: ObservationRef,
        current_observation: Observation,
        final_action: ActionSpec,
    ) -> PostDecisionPredictions:
        frame = normalize_frame_for_memory(current_observation.frame)
        common_metadata = {
            "purpose": "post_decision_update_prediction",
            "prompt_model_calls_enabled": False,
        }
        return PostDecisionPredictions(
            world_prediction=ToolResult(
                id=f"post-decision-world-{current_observation.id}",
                tool="world",
                predicted_observation=frame,
                source_observation_ref=current_observation_ref,
                action=final_action,
                explanation="Mock post-decision world prediction.",
                metadata=dict(common_metadata),
            ),
            goal_prediction=ToolResult(
                id=f"post-decision-goal-{current_observation.id}",
                tool="goal",
                predicted_observation=frame,
                source_observation_ref=current_observation_ref,
                explanation="Mock post-decision goal prediction.",
                metadata=dict(common_metadata),
            ),
        )
