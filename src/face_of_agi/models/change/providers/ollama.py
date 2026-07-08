"""Ollama provider for transition change summaries."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.change.contracts import ChangeSummaryProviderResponse
from face_of_agi.models.image_inputs import ollama_image_payloads
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    OllamaStructuredChatResult,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)


class OllamaChangeSummaryProvider:
    """Final Ollama chat transport for transition change summaries."""

    backend = "ollama"

    def __init__(
        self,
        config: Any,
        *,
        client: Any | None = None,
    ) -> None:
        if not getattr(config, "model", None):
            raise ValueError("Ollama change summary requires an explicit model")
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
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
        """Call Ollama and return raw change summary JSON text."""

        result = self._client.structured_chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=prompt_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
            response_format=output_schema,
        )
        self._capture_calls(result, phase="complete")
        return self._provider_response(result.response)

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
        """Ask Ollama to repair invalid change summary JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous change summary was invalid.",
                "Original request:\n" + prompt_text,
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Return only corrected JSON with array field 'elements' and "
                "boolean field 'change_detected'.",
            ]
        )
        result = self._client.structured_chat(
            model=self.config.model,
            messages=self._messages(
                instructions_text=instructions_text,
                prompt_text=repair_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
            response_format=output_schema,
        )
        self._capture_calls(
            result,
            phase="repair_complete",
            attempt=attempt,
        )
        return self._provider_response(result.response)

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
            {"role": "system", "content": instructions_text},
            self._user_message(
                prompt_text,
                previous_image=previous_image,
                current_image=current_image,
                images=images,
            ),
            assistant_json_prefill_message(),
        ]

    def _user_message(
        self,
        content: str,
        *,
        previous_image: Any,
        current_image: Any,
        images: Any | None,
    ) -> dict[str, Any]:
        image_sequence = tuple(images or (previous_image, current_image))
        return {
            "role": "user",
            "content": content,
            "images": ollama_image_payloads(
                image_sequence,
                size=None,
                resample=self.config.input_image_resample,
            ),
        }

    def _provider_response(self, response: Any) -> ChangeSummaryProviderResponse:
        text = structured_json_content(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            "usage": response_usage(response),
        }
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = metadata
        return ChangeSummaryProviderResponse(
            text=text,
            request=self.last_request,
            metadata=metadata,
        )

    def _capture_calls(
        self,
        result: OllamaStructuredChatResult,
        *,
        phase: str,
        attempt: int | None = None,
    ) -> None:
        for call in result.calls:
            call_phase = f"{phase}_thinking" if call.kind == "thinking" else phase
            capture_ollama_model_input(
                self,
                call_slot="change",
                provider=self.backend,
                model=self.model,
                phase=call_phase,
                attempt=attempt,
                request=call.request,
                response=call.response,
            )
