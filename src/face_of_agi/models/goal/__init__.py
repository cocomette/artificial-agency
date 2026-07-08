"""Goal prediction model G package."""

from face_of_agi.models.description import (
    OllamaDescriptionConfig,
    OpenAIDescriptionConfig,
)
from face_of_agi.models.goal.adapter import (
    GOAL_DESCRIPTION_ROLE,
    GoalPredictionAdapter,
)
from face_of_agi.models.goal.contracts import GoalPredictionModel

__all__ = [
    "GOAL_DESCRIPTION_ROLE",
    "GoalPredictionAdapter",
    "GoalPredictionModel",
    "OllamaDescriptionConfig",
    "OpenAIDescriptionConfig",
]
