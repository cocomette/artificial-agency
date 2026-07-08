"""Contracts for the Interest/value model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from face_of_agi.contracts import (
    ActionHistoryItem,
    AgentCandidateAction,
    GoalPrediction,
    InterestPrediction,
    MemoryDocument,
    Observation,
    WorldPrediction,
)


def interest_prediction_json_schema() -> dict[str, Any]:
    """Return the Interest output schema."""

    return {
        "type": "object",
        "properties": {
            "candidate_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "expected_learning_progress": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                        },
                        "expected_goal_delta": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "candidate_index",
                        "expected_learning_progress",
                        "expected_goal_delta",
                        "confidence",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["candidate_values"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class InterestPredictionInput:
    """Input for scoring a full candidate set."""

    run_id: str
    game_id: str
    turn_id: int
    current_observation: Observation
    memory: MemoryDocument
    goal: GoalPrediction
    candidates: Sequence[AgentCandidateAction]
    world_predictions: Sequence[WorldPrediction]
    recent_action_history: tuple[ActionHistoryItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class InterestModel(Protocol):
    """Model role that predicts candidate-level curiosity and goal value."""

    def score_candidates(
        self,
        prediction_input: InterestPredictionInput,
    ) -> InterestPrediction:
        """Return one value estimate per candidate."""
        ...

