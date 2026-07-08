"""vLLM provider for transition change summaries."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.change.contracts import ChangeSummaryProviderResponse
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)


class VLLMChangeSummaryProvider:
    """Final vLLM chat transport for transition change summaries."""

    backend = "vllm"

    def __init__(
        self,
        config: Any,
        *,
        client: Any | None = None,
    ) -> None:
        if not getattr(config, "model", None):
            raise ValueError("vLLM change summary requires an explicit model")
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        previous_image: Any,
        current_image: Any,
        output_schema: dict[str, Any],
        images: Any | None = None,
    ) -> ChangeSummaryProviderResponse:
        """Call vLLM and return raw change summary JSON text."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=prompt_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
            response_format=json_schema_response_format(
                name="change_summary",
                schema=output_schema,
            ),
        )
        self._capture_request(phase="complete", response=response)
        return self._provider_response(response)

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        previous_image: Any,
        current_image: Any,
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
        images: Any | None = None,
    ) -> ChangeSummaryProviderResponse:
        """Ask vLLM to repair invalid change summary JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous change summary was invalid.",
                "Original request:\n" + prompt_text,
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Return only corrected JSON with non-empty string 'summary' and "
                "boolean 'change_detected' fields.",
            ]
        )
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=repair_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
            response_format=json_schema_response_format(
                name="change_summary",
                schema=output_schema,
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
        previous_image: Any,
        current_image: Any,
        images: Any | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": instructions_text,
            },
            self._user_message(
                prompt_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
        ]

    def _user_message(
        self,
        prompt: str,
        *,
        previous_image: Any,
        current_image: Any,
        images: Any | None,
    ) -> dict[str, Any]:
        attached_images = tuple(images or (previous_image, current_image))
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *vllm_image_content(
                    attached_images,
                    detail=self.config.input_image_detail,
                    size=None,
                    resample=self.config.input_image_resample,
                    mime_type=self.config.image_mime_type,
                ),
            ],
        }

    def _provider_response(self, response: Any) -> ChangeSummaryProviderResponse:
        text = chat_message_optional_content(response) or ""
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **chat_response_metadata(response),
        }
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = metadata
        return ChangeSummaryProviderResponse(
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
            call_slot="change",
            provider=self.backend,
            model=self.model,
            phase=phase,
            attempt=attempt,
            request=request,
            response=response,
            metadata={
                "response_metadata": chat_response_metadata(response),
            },
        )
