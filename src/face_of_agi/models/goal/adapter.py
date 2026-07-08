"""Goal prediction model G adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_of_agi.contracts import Observation, PredictionResult, RoleContext
from face_of_agi.models.description import (
    DescriptionPredictionAdapter,
    DescriptionProvider,
    DescriptionRoleSpec,
)

GOAL_DESCRIPTION_ROLE = DescriptionRoleSpec(
    tool_name="goal",
    id_prefix="goal",
    instruction_dir=Path(__file__).parent / "instructions",
    validation_label="goal description prediction",
    provider_label="goal",
    explanation="Predicted goal-relevant state as a structured description.",
    include_action=False,
)


class GoalPredictionAdapter:
    """Goal model G adapter with provider-neutral prompt/result logic."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        client: Any | None = None,
        provider: DescriptionProvider | None = None,
    ) -> None:
        self._adapter = DescriptionPredictionAdapter(
            role=GOAL_DESCRIPTION_ROLE,
            config=config,
            client=client,
            provider=provider,
        )

    @property
    def config(self) -> Any:
        return self._adapter.config

    @property
    def provider(self) -> DescriptionProvider:
        return self._adapter.provider

    @property
    def last_instructions(self) -> str | None:
        return self._adapter.last_instructions

    @property
    def last_prompt(self) -> str | None:
        return self._adapter.last_prompt

    @property
    def last_request(self) -> dict[str, Any] | None:
        return self._adapter.last_request

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> PredictionResult:
        """Predict goal-relevant visual state as a structured description."""

        return self._adapter.predict(context, observation)
