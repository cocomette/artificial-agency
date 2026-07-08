"""vLLM provider for structured description predictions."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.description.config import openai_description_response_schema
from face_of_agi.models.description.contracts import (
    DescriptionProviderResponse,
    DescriptionRoleSpec,
)
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.providers.vision import resolve_model_vision_profile


class VLLMDescriptionProvider:
    """Final vLLM chat transport for description prediction prompts."""

    backend = "vllm"

    def __init__(
        self,
        config: Any,
        *,
        role: DescriptionRoleSpec,
        client: Any | None = None,
    ) -> None:
        if not getattr(config, "model", None):
            raise ValueError(
                f"vLLM {role.provider_label} prediction requires an explicit model"
            )
        self.config = config
        self.role = role
        self._client = VLLMChatClient(config, client=client)
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
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=prompt_text,
                image=image,
            ),
            response_format=json_schema_response_format(
                name="description_prediction",
                schema=openai_description_response_schema(),
            ),
        )
        self._capture_request(phase="complete", response=response)
        return self._provider_response(response)

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> DescriptionProviderResponse:
        """Ask vLLM to repair invalid description JSON."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._repair_messages(
                instructions_text=instructions_text,
                prompt_text=prompt_text,
                image=image,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            ),
            response_format=json_schema_response_format(
                name="description_prediction",
                schema=openai_description_response_schema(),
            ),
        )
        self._capture_request(
            phase="repair_complete",
            response=response,
            attempt=attempt,
        )
        return self._provider_response(response)

    def _messages(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": instructions_text,
            },
            self._user_message(prompt_text, image=image),
        ]

    def _repair_messages(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        image: Any | None,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                (
                    f"Repair attempt {attempt}: the previous "
                    f"{self.role.provider_label} prediction output was invalid."
                ),
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                f"Original {self.role.provider_label} prediction user prompt:\n"
                + prompt_text,
                "Return only corrected JSON that validates against the schema.",
            ]
        )
        return [
            {
                "role": "system",
                "content": instructions_text,
            },
            self._user_message(repair_text, image=image),
        ]

    def _provider_response(self, response: Any) -> DescriptionProviderResponse:
        text = chat_message_optional_content(response) or ""
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **chat_response_metadata(response),
            "visual_coordinate_space": self.coordinate_space,
            "visual_coordinate_space_source": self.coordinate_space_source,
        }
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = metadata
        return DescriptionProviderResponse(
            text=text,
            request=self.last_request,
            metadata=metadata,
        )

    def _capture_request(
        self,
        *,
        phase: str,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        request = self._client.last_request
        if request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot=self.role.tool_name,
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=request,
            response=response,
            metadata={
                "visual_coordinate_space": self.coordinate_space,
                "visual_coordinate_space_source": self.coordinate_space_source,
            },
        )

    def _user_message(self, content: str, *, image: Any | None) -> dict[str, Any]:
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": content,
            }
        ]
        if image is not None:
            message_content.extend(
                vllm_image_content(
                    (image,),
                    detail=self.config.input_image_detail,
                    size=self.config.input_image_size,
                    resample=self.config.input_image_resample,
                    mime_type=self.config.image_mime_type,
                )
            )
        return {"role": "user", "content": message_content}
