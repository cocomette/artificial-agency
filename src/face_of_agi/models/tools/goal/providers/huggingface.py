"""Hugging Face/Diffusers provider for the goal model tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from face_of_agi.contracts import (
    Observation,
    ObservationRef,
    RoleContext,
    ToolResult,
)
from face_of_agi.models.providers.huggingface import DiffusersImageEditorAdapter
from face_of_agi.models.tools.goal.config import GoalToolConfig


class HuggingFaceGoalToolAdapter(DiffusersImageEditorAdapter):
    """Predict goal-relevant observations with a local image-editing backend."""

    def __init__(
        self,
        config: GoalToolConfig | None = None,
        *,
        pipeline: Any | None = None,
    ) -> None:
        super().__init__(
            config or GoalToolConfig(),
            pipeline=pipeline,
            prompt_dir=Path(__file__).parent.parent / "instructions",
            role_name="goal",
        )

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        """Predict a goal-relevant observation for the supplied observation."""

        output_image = self._predict_image(context, observation)

        return ToolResult(
            id=f"goal-{uuid4().hex}",
            tool="goal",
            predicted_observation=output_image,
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            explanation=(
                "Predicted a goal-relevant observation with a Diffusers image editor "
                "from the provided observation and goal context."
            ),
            metadata=self._metadata(output_image.size),
        )

    def _compose_prompt(
        self,
        context: RoleContext,
        observation: Observation,
        action: object | None = None,
    ) -> str:
        """Build the image-edit instruction sent to the goal image editor."""

        if self.config.pipeline_type == "instruct_pix2pix":
            return self._compose_instruct_pix2pix_prompt(context)

        if self.config.pipeline_type == "flux_kontext_qint8":
            return self._compose_flux_kontext_prompt(context)

        return self._compose_qwen_image_edit_prompt(context, observation)

    def _compose_qwen_image_edit_prompt(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> str:
        """Build the structured prompt for long-context image editors."""

        context_text = context.composed().strip()
        if not context_text:
            context_text = "(no game-specific goal context supplied)"

        return "\n\n".join(
            [
                self._load_instruction_prompt(),
                "GOAL MODEL DOC (K^G + L^G):\n" + context_text,
                "SOURCE OBSERVATION:\n"
                f"id: {observation.id}\n"
                f"step: {observation.step}\n"
                f"frame_count: {observation.frame_count()}",
            ]
        )

    def _compose_instruct_pix2pix_prompt(
        self,
        context: RoleContext,
    ) -> str:
        """Build a compact CLIP-length prompt for InstructPix2Pix."""

        parts = [
            self._load_instruction_prompt("instruction_prompt_instruct_pix2pix.md"),
        ]
        goal_hint = self._compact_goal_hint(context)
        if goal_hint:
            parts.append(f"Goal hint: {goal_hint}")
        return " ".join(parts)

    def _compose_flux_kontext_prompt(
        self,
        context: RoleContext,
    ) -> str:
        """Build a compact goal prompt for FLUX Kontext tokenizers."""

        parts = [
            self._load_instruction_prompt("instruction_prompt_flux_kontext.md"),
        ]
        goal_hint = self._compact_goal_hint(context, max_chars=220)
        if goal_hint:
            parts.append(f"Goal model doc: {goal_hint}")
        return " ".join(parts)

    def _compact_goal_hint(self, context: RoleContext, max_chars: int = 96) -> str:
        """Return a short goal-doc hint that fits compact prompts."""

        return self._compact_context_hint(context, max_chars=max_chars)
