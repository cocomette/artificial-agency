"""Contracts for the World model role."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from face_of_agi.contracts import ActionSpec, MemoryDocument, Observation, WorldPrediction


def world_prediction_json_schema() -> dict[str, Any]:
    """Return the World output schema."""

    return {
        "type": "object",
        "properties": {
            "predicted_change": {
                "type": "string",
                "description": "Change-summary-style text prediction.",
            },
        },
        "required": ["predicted_change"],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class WorldPredictionInput:
    """Input for predicting one candidate transition."""

    run_id: str
    game_id: str
    candidate_index: int
    current_observation: Observation
    action: ActionSpec
    memory: MemoryDocument
    glossary_actions: Sequence[ActionSpec] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class WorldModel(Protocol):
    """Model role that predicts transition text for candidate actions."""

    def predict_transition(
        self,
        prediction_input: WorldPredictionInput,
    ) -> WorldPrediction:
        """Return one transition prediction."""
        ...
