"""Contracts for the goal prediction model."""

from __future__ import annotations

from typing import Protocol

from face_of_agi.contracts import Observation, PredictionResult, RoleContext


class GoalPredictionModel(Protocol):
    """Goal prediction role G."""

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> PredictionResult:
        """Predict or evaluate goal-relevant observations."""
        ...
