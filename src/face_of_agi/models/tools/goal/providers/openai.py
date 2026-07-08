"""OpenAI Responses adapter for the goal model tool."""

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
from face_of_agi.models.providers.openai import OpenAIImageGenerationClient
from face_of_agi.models.tools.goal.config import OpenAIGoalToolConfig


class OpenAIGoalToolAdapter:
    """Predict goal-relevant observations through OpenAI image generation."""

    def __init__(
        self,
        config: OpenAIGoalToolConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or OpenAIGoalToolConfig()
        self._openai = OpenAIImageGenerationClient(self.config, client=client)
        self._prompt_dir = Path(__file__).parent.parent / "instructions"
        self._instruction_prompts: dict[str, str] = {}
        self.last_prompt: str | None = None

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> ToolResult:
        """Predict a goal-relevant observation for the supplied observation."""

        prompt = self._compose_prompt(context, observation)
        self.last_prompt = prompt
        result = self._openai.generate_image(prompt=prompt, observation=observation)

        return ToolResult(
            id=f"goal-{uuid4().hex}",
            tool="goal",
            predicted_observation=result.image,
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            explanation=(
                result.output_text
                or "Predicted goal-relevant observation with OpenAI Responses image generation."
            ),
            metadata=self._metadata(result.metadata, result.image.size),
        )

    def _compose_prompt(
        self,
        context: RoleContext,
        observation: Observation,
    ) -> str:
        """Build the multimodal prompt sent to OpenAI."""

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

    def _metadata(
        self,
        response_metadata: dict[str, Any],
        image_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return goal-tool OpenAI metadata for a ToolResult."""

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
        """Read one fixed goal instruction prompt once."""

        if filename not in self._instruction_prompts:
            prompt_path = self._prompt_dir / filename
            self._instruction_prompts[filename] = prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        return self._instruction_prompts[filename]
