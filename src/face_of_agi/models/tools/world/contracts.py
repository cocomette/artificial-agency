"""Contracts for the world model tool."""

from __future__ import annotations

from typing import Protocol

from face_of_agi.contracts import ActionSpec, Observation, RoleContext, ToolResult


class WorldToolModel(Protocol):
    """World-model tool role S."""

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        """Predict the next observation for an action."""
        ...
