"""Contracts for the goal model tool."""

from __future__ import annotations

from typing import Protocol

from face_of_agi.contracts import Observation, RoleContext, ToolResult


class GoalToolModel(Protocol):
    """Goal-model tool role G."""

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        """Predict or evaluate goal-relevant observations."""
        ...
