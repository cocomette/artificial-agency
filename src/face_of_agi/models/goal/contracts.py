"""Contracts for the Goal role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import GoalPrediction, MemoryDocument, Observation


def goal_output_json_schema() -> dict[str, Any]:
    """Return the Goal role output schema."""

    return {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "subgoals": {
                "type": "array",
                "items": {"type": "string"},
            },
            "steps_remaining": {
                "type": "integer",
                "minimum": 0,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["goal", "subgoals", "steps_remaining", "confidence"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class GoalPredictionInput:
    """Input for one Goal model call."""

    run_id: str
    game_id: str
    memory: MemoryDocument
    current_observation: Observation
    previous_goal: GoalPrediction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalModel(Protocol):
    """Model role that predicts the current goal and remaining steps."""

    def predict_goal(self, prediction_input: GoalPredictionInput) -> GoalPrediction:
        """Return a structured goal prediction."""
        ...
