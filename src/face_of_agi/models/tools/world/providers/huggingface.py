"""Hugging Face/Diffusers provider for the world model tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from face_of_agi.contracts import (
    ActionSpec,
    Observation,
    ObservationRef,
    RoleContext,
    ToolResult,
)
from face_of_agi.models.providers.huggingface import DiffusersImageEditorAdapter
from face_of_agi.models.tools.world.config import WorldToolConfig


class HuggingFaceWorldToolAdapter(DiffusersImageEditorAdapter):
    """Predict next observations with a local image-editing backend."""

    def __init__(
        self,
        config: WorldToolConfig | None = None,
        *,
        pipeline: Any | None = None,
    ) -> None:
        super().__init__(
            config or WorldToolConfig(),
            pipeline=pipeline,
            prompt_dir=Path(__file__).parent.parent / "instructions",
            role_name="world",
        )

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        """Predict the next visual observation for one proposed action."""

        output_image = self._predict_image(context, observation, action)

        return ToolResult(
            id=f"world-{uuid4().hex}",
            tool="world",
            predicted_observation=output_image,
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
            explanation=(
                "Predicted next observation with a Diffusers image editor from the "
                "provided observation and proposed action."
            ),
            metadata=self._metadata(output_image.size),
        )

    def _compose_prompt(
        self,
        context: RoleContext,
        action: ActionSpec | None = None,
        observation: Observation | None = None,
    ) -> str:
        """Build the image-edit instruction sent to the world image editor."""

        if action is None:
            raise ValueError("world model predictions require a proposed action")
        if observation is None:
            raise ValueError("world model predictions require a source observation")

        if self.config.pipeline_type == "instruct_pix2pix":
            return self._compose_instruct_pix2pix_prompt(context, action)

        if self.config.pipeline_type == "flux_kontext_qint8":
            return self._compose_flux_kontext_prompt(context, action)

        return self._compose_qwen_image_edit_prompt(context, action, observation)

    def _compose_qwen_image_edit_prompt(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> str:
        """Build the structured prompt for long-context image editors."""

        context_text = context.composed().strip()
        if not context_text:
            context_text = "(no game-specific world context supplied)"

        return "\n\n".join(
            [
                self._load_instruction_prompt(),
                "WORLD MODEL DOC (K^S + L^S):\n" + context_text,
                "SOURCE OBSERVATION:\n"
                f"id: {observation.id}\n"
                f"step: {observation.step}\n"
                f"frame_count: {observation.frame_count()}",
                "PROPOSED ACTION:\n"
                f"action_id: {self._action_id_text(action)}\n"
                f"data: {self._action_data_text(action)}",
            ]
        )

    def _compose_instruct_pix2pix_prompt(
        self,
        context: RoleContext,
        action: ActionSpec,
    ) -> str:
        """Build a compact CLIP-length prompt for InstructPix2Pix."""

        parts = [
            f"Action {self._compact_action_text(action)}.",
            self._load_instruction_prompt("instruction_prompt_instruct_pix2pix.md"),
        ]
        world_hint = self._compact_world_hint(context)
        if world_hint:
            parts.append(f"Hint: {world_hint}")
        return " ".join(parts)

    def _compose_flux_kontext_prompt(
        self,
        context: RoleContext,
        action: ActionSpec,
    ) -> str:
        """Build an action-first prompt for FLUX Kontext tokenizers."""

        parts = [
            f"Action {self._compact_action_text(action)}.",
            self._load_instruction_prompt("instruction_prompt_flux_kontext.md"),
        ]
        world_hint = self._compact_world_hint(context, max_chars=220)
        if world_hint:
            parts.append(f"World model doc: {world_hint}")
        return " ".join(parts)

    def _compact_world_hint(self, context: RoleContext, max_chars: int = 96) -> str:
        """Return a short world-doc hint that fits compact prompts."""

        return self._compact_context_hint(context, max_chars=max_chars)
