"""Contracts for the world prediction model."""

from __future__ import annotations

from typing import Protocol

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    PredictionResult,
    RoleContext,
)


class WorldPredictionModel(Protocol):
    """World prediction role S."""

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> PredictionResult:
        """Predict the next observation for an action."""
        ...
