"""OpenAI provider for structured description predictions."""

from __future__ import annotations

from typing import Any

from face_of_agi.models.description.config import openai_description_text_format
from face_of_agi.models.description.contracts import (
    DescriptionProviderResponse,
    DescriptionRoleSpec,
)
from face_of_agi.models.image_inputs import openai_image_content
from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)
from face_of_agi.models.providers.vision import resolve_model_vision_profile


class OpenAIDescriptionProvider:
    """Final OpenAI Responses transport for description prediction prompts."""

    def __init__(
        self,
        config: Any,
        *,
        role: DescriptionRoleSpec,
        client: Any | None = None,
    ) -> None:
        config.text = {
            **(getattr(config, "text", None) or {}),
            **openai_description_text_format(),
        }
        self.config = config
        self.role = role
        self._client = OpenAIResponsesClient(config, client=client)
        profile = resolve_model_vision_profile(
            backend=config.backend,
            model=config.model,
        )
        self.coordinate_space = profile.coordinate_space
        self.coordinate_space_source = profile.source
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
    ) -> DescriptionProviderResponse:
        response = self._client.create_response(
            model=self.config.model,
            instructions=self._instructions(instructions_text),
            input_items=[self._input_item(prompt_text, image=image)],
        )
        self.last_request = self._client.last_request
        self._capture_request(phase="complete", response=response)
        output_text = response_output_text(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **openai_response_metadata(response),
            "visual_coordinate_space": self.coordinate_space,
            "visual_coordinate_space_source": self.coordinate_space_source,
        }
        self.last_response_text = output_text
        self.last_response_metadata = metadata
        if output_text is None:
            raise RuntimeError(
                f"OpenAI {self.role.provider_label} prediction response did not "
                "include output text"
            )
        return DescriptionProviderResponse(
            text=output_text,
            request=self.last_request,
            metadata=metadata,
        )

    def _capture_request(self, *, phase: str, response: Any | None) -> None:
        request = self._client.last_request
        if request is None:
            return
        capture_openai_model_input(
            self,
            call_slot=self.role.tool_name,
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=request,
            response=response,
            metadata={
                "visual_coordinate_space": self.coordinate_space,
                "visual_coordinate_space_source": self.coordinate_space_source,
            },
        )

    def _input_item(self, prompt: str, *, image: Any | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if image is not None:
            content.extend(
                openai_image_content(
                    (image,),
                    detail=self.config.input_image_detail,
                    size=self.config.input_image_size,
                    resample=self.config.input_image_resample,
                    mime_type=self.config.image_mime_type,
                )
            )
        return {"role": "user", "content": content}

    def _instructions(self, instructions_text: str) -> str:
        return "\n\n".join(
            block.strip()
            for block in (
                getattr(self.config, "instructions", None) or "",
                instructions_text,
            )
            if block.strip()
        )
