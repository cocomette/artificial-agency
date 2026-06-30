"""vLLM provider for transition change summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.change.contracts import ChangeSummaryProviderResponse
from face_of_agi.models.image_inputs import vllm_text_image_content
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.structured_output import (
    DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS,
    clipped_invalid_output_preview,
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
        images: Sequence[Any],
        output_schema: dict[str, Any],
    ) -> ChangeSummaryProviderResponse:
        """Call vLLM and return raw change summary JSON text."""

        return self._complete(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            images=images,
            output_schema=output_schema,
            phase="complete",
        )

    def repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> ChangeSummaryProviderResponse:
        """Ask vLLM to repair invalid change summary JSON."""

        return self._repair_complete(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            images=images,
            output_schema=output_schema,
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
            phase="repair_complete",
            description="the previous change summary was invalid",
        )

    def reduce_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
    ) -> ChangeSummaryProviderResponse:
        """Call vLLM and return raw reduced change summary JSON text."""

        return self._complete(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            images=images,
            output_schema=output_schema,
            phase="reduce_complete",
        )

    def repair_reduce_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> ChangeSummaryProviderResponse:
        """Ask vLLM to repair invalid reduced change summary JSON."""

        return self._repair_complete(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            images=images,
            output_schema=output_schema,
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
            phase="repair_reduce_complete",
            description="the previous reduced change summary was invalid",
        )

    def _complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        phase: str,
    ) -> ChangeSummaryProviderResponse:
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=prompt_text,
                images=images,
            ),
            response_format=json_schema_response_format(
                name="change_summary",
                schema=output_schema,
            ),
        )
        self._capture_request(phase=phase, response=response)
        return self._provider_response(response)

    def _repair_complete(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
        output_schema: dict[str, Any],
        invalid_text: str,
        validation_error: str,
        attempt: int,
        phase: str,
        description: str,
    ) -> ChangeSummaryProviderResponse:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: {description}.",
                "Original request:\n" + prompt_text,
                "Validation error:\n" + validation_error,
                "Invalid output preview:\n"
                + clipped_invalid_output_preview(
                    invalid_text,
                    max_chars=getattr(
                        self.config,
                        "repair_invalid_output_preview_chars",
                        DEFAULT_INVALID_OUTPUT_PREVIEW_CHARS,
                    ),
                ),
                "Return only corrected JSON with a non-empty string `summary` "
                "field and boolean `change_detected` field.",
            ]
        )
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=repair_text,
                images=images,
            ),
            response_format=json_schema_response_format(
                name="change_summary",
                schema=output_schema,
            ),
        )
        self._capture_request(
            phase=phase,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(response)

    def _messages(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        images: Sequence[Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": instructions_text,
            },
            self._user_message(prompt_text, images=images),
        ]

    def _user_message(self, prompt: str, *, images: Sequence[Any]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": vllm_text_image_content(
                prompt,
                images,
                detail=self.config.input_image_detail,
                mime_type=self.config.image_mime_type,
            ),
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
