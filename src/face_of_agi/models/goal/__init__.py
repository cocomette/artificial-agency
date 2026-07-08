"""Goal model role."""

from face_of_agi.models.goal.adapter import VLLMGoalAdapter
from face_of_agi.models.goal.config import VLLMGoalConfig
from face_of_agi.models.goal.contracts import (
    GoalModel,
    GoalPredictionInput,
    goal_output_json_schema,
)

__all__ = [
    "GoalModel",
    "GoalPredictionInput",
    "VLLMGoalAdapter",
    "VLLMGoalConfig",
    "goal_output_json_schema",
]
