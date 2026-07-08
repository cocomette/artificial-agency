"""OpenAI Responses adapter for the world model tool."""

from __future__ import annotations

import json
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
from face_of_agi.models.providers.openai import OpenAIImageGenerationClient
from face_of_agi.models.tools.world.config import OpenAIWorldToolConfig


class OpenAIWorldToolAdapter:
    """Predict next observations through OpenAI Responses image generation."""

    def __init__(
        self,
        config: OpenAIWorldToolConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or OpenAIWorldToolConfig()
        self._openai = OpenAIImageGenerationClient(self.config, client=client)
        self._prompt_dir = Path(__file__).parent.parent / "instructions"
        self._instruction_prompts: dict[str, str] = {}
        self.last_prompt: str | None = None

    def predict(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> ToolResult:
        """Predict the next visual observation for one proposed action."""

        prompt = self._compose_prompt(context, action, observation)
        self.last_prompt = prompt
        result = self._openai.generate_image(prompt=prompt, observation=observation)

        return ToolResult(
            id=f"world-{uuid4().hex}",
            tool="world",
            predicted_observation=result.image,
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action,
            explanation=(
                result.output_text
                or "Predicted next observation with OpenAI Responses image generation."
            ),
            metadata=self._metadata(result.metadata, result.image.size),
        )

    def _compose_prompt(
        self,
        context: RoleContext,
        action: ActionSpec,
        observation: Observation,
    ) -> str:
        """Build the multimodal prompt sent to OpenAI."""

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

    def _metadata(
        self,
        response_metadata: dict[str, Any],
        image_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return world-tool OpenAI metadata for a ToolResult."""

        return {
            **response_metadata,
            "image_action": self.config.image_action,
            "image_quality": self.config.image_quality,
            "image_size": image_size,
            "image_output_format": self.config.image_output_format,
            "input_image_detail": self.config.input_image_detail,
            "input_image_size": self.config.input_image_size,
            "input_image_resample": self.config.input_image_resample,
            "max_tool_calls": self.config.max_tool_calls,
            "reasoning": self.config.reasoning,
            "tool_choice": "image_generation",
        }

    def _load_instruction_prompt(self, filename: str = "instruction_prompt.md") -> str:
        """Read one fixed world instruction prompt once."""

        if filename not in self._instruction_prompts:
            prompt_path = self._prompt_dir / filename
            self._instruction_prompts[filename] = prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        return self._instruction_prompts[filename]

    def _action_id_text(self, action: ActionSpec) -> str:
        """Return a compact action id for prompts and logs."""

        return str(getattr(action.action_id, "name", action.action_id))

    def _action_data_text(self, action: ActionSpec) -> str:
        """Render action payloads deterministically for the model prompt."""

        if action.data is None:
            return "{}"
        return json.dumps(action.data, sort_keys=True)
