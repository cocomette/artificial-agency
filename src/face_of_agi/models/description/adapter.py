"""Provider-neutral adapter for S/G structured description predictions."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from face_of_agi.contracts import (
    ActionSpec,
    DESCRIPTION_SCHEMA,
    DescriptionPrediction,
    DescriptionPredictionError,
    Observation,
    ObservationRef,
    PredictionResult,
    RoleContext,
    parse_description_prediction,
)
from face_of_agi.frames import observation_to_pil_image
from face_of_agi.models.description.config import OllamaDescriptionConfig
from face_of_agi.models.description.contracts import (
    DescriptionProvider,
    DescriptionRoleSpec,
)
from face_of_agi.models.structured_output import (
    append_output_schema_to_instructions,
    provider_repair_callback,
    validate_with_repair,
)


class DescriptionPredictionAdapter:
    """Shared prompt/result logic for world and goal description roles."""

    def __init__(
        self,
        *,
        role: DescriptionRoleSpec,
        config: Any | None = None,
        client: Any | None = None,
        provider: DescriptionProvider | None = None,
    ) -> None:
        self.role = role
        self.config = config or OllamaDescriptionConfig()
        self.provider = provider or self._default_provider(client=client)
        self.last_instructions: str | None = None
        self.last_prompt: str | None = None
        self.last_request: dict[str, Any] | None = None

    def predict(
        self,
        context: RoleContext,
        observation: Observation,
        *,
        action: ActionSpec | None = None,
    ) -> PredictionResult:
        """Predict structured visual descriptions for one S/G role."""

        if self.role.include_action and action is None:
            raise ValueError(f"{self.role.tool_name} description prediction requires an action")
        instructions_text = append_output_schema_to_instructions(
            _load_instruction_prompt(self.role),
            _description_output_schema(self.config),
            include=bool(
                getattr(self.config, "include_output_schema_in_instructions", False)
            ),
        )
        prompt_text = self._compose_user_prompt(
            context=context,
            action=action,
        )
        self.last_instructions = instructions_text
        self.last_prompt = prompt_text
        image = observation_to_pil_image(
            observation,
            frame_scale=getattr(self.config, "frame_scale", 4),
        )
        response = self.provider.complete(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            image=image,
        )
        self.last_request = response.request
        validated = validate_with_repair(
            label=self.role.validation_label,
            response=response,
            text_of=lambda item: item.text,
            validate=lambda text: parse_provider_description(
                text,
                image=image,
                provider=self.provider,
            ),
            repair=provider_repair_callback(
                self.provider,
                "repair_complete",
                kwargs={
                    "instructions_text": instructions_text,
                    "prompt_text": prompt_text,
                    "image": image,
                },
            ),
            max_repair_attempts=getattr(self.config, "repair_attempts", 0),
            error_factory=DescriptionPredictionError,
        )
        response = validated.response
        self.last_request = response.request
        return PredictionResult(
            id=f"{self.role.id_prefix}-{uuid4().hex}",
            tool=self.role.tool_name,
            predicted_description=validated.value,
            source_observation_ref=ObservationRef(memory="state", id=observation.id),
            action=action if self.role.include_action else None,
            explanation=self.role.explanation,
            metadata={
                **response.metadata,
                "input_source": "image",
                "repair_attempts": validated.repair_attempts,
            },
        )

    def _default_provider(self, *, client: Any | None) -> DescriptionProvider:
        backend = (getattr(self.config, "backend", None) or "").lower()
        if backend == "openai":
            from face_of_agi.models.description.providers.openai import (
                OpenAIDescriptionProvider,
            )

            return OpenAIDescriptionProvider(
                self.config,
                role=self.role,
                client=client,
            )
        if backend == "ollama":
            from face_of_agi.models.description.providers.ollama import (
                OllamaDescriptionProvider,
            )

            return OllamaDescriptionProvider(
                self.config,
                role=self.role,
                client=client,
            )
        if backend == "vllm":
            from face_of_agi.models.description.providers.vllm import (
                VLLMDescriptionProvider,
            )

            return VLLMDescriptionProvider(
                self.config,
                role=self.role,
                client=client,
            )
        raise ValueError(f"unknown {self.role.tool_name} backend: {self.config.backend}")

    def _compose_user_prompt(
        self,
        *,
        context: RoleContext,
        action: ActionSpec | None,
    ) -> str:
        blocks = [
            _role_context_block(context),
        ]
        if self.role.include_action:
            assert action is not None
            blocks.append(
                "ACTION:\n"
                f"action_id: {action.name}\n"
                f"data: {_action_data_text(action)}"
            )
        return "\n\n".join(blocks)


def parse_provider_description(
    text: str,
    *,
    image: Any | None,
    provider: object,
) -> DescriptionPrediction:
    """Parse provider description text using the provider coordinate convention."""

    return parse_description_prediction(
        text,
        image_size=image.size if image is not None else None,
        coordinate_space=getattr(provider, "coordinate_space", "pixel"),
    )


def _load_instruction_prompt(role: DescriptionRoleSpec) -> str:
    return (role.instruction_dir / "instruction_prompt.md").read_text(
        encoding="utf-8"
    ).strip()


def _description_output_schema(config: Any) -> dict[str, Any]:
    text = getattr(config, "text", None)
    if isinstance(text, dict):
        text_format = text.get("format")
        if isinstance(text_format, dict):
            schema = text_format.get("schema")
            if isinstance(schema, dict):
                return schema

    response_format = getattr(config, "format", None)
    if isinstance(response_format, dict):
        return response_format
    return DESCRIPTION_SCHEMA


def _context_text(context: RoleContext) -> str:
    return context.composed().strip()


def _role_context_block(context: RoleContext) -> str:
    text = _context_text(context)
    if not text:
        return "ROLE_CONTEXT:"
    return "ROLE_CONTEXT:\n" + text


def _action_data_text(action: ActionSpec) -> str:
    if action.data is None:
        return "{}"
    return json.dumps(action.data, sort_keys=True)
