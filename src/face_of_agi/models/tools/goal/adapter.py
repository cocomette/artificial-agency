"""Framework adapter for goal model tool G."""

from __future__ import annotations

from face_of_agi.models.tools.goal.providers.huggingface import (
    HuggingFaceGoalToolAdapter,
)


class GoalToolAdapter(HuggingFaceGoalToolAdapter):
    """Default goal tool adapter backed by Hugging Face/Diffusers."""
