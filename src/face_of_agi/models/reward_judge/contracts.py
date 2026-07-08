"""Contracts for the Reward Judge role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from face_of_agi.contracts import ActionSpec, Observation, RewardJudgeScore, WorldPrediction


def reward_judge_json_schema() -> dict[str, Any]:
    """Return the Reward Judge output schema."""

    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "string"},
            "error_tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["score", "notes", "error_tags"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class RewardJudgeInput:
    """Input for judging a world prediction against ground truth."""

    run_id: str
    game_id: str
    turn_id: int
    action: ActionSpec
    prediction: WorldPrediction
    change_summary: str
    previous_observation: Observation
    current_observation: Observation
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardJudgeModel(Protocol):
    """Model role that scores text prediction quality."""

    def judge_prediction(self, judge_input: RewardJudgeInput) -> RewardJudgeScore:
        """Return a scalar quality score and notes."""
        ...
