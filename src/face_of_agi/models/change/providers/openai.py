"""OpenAI provider for transition change summaries."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.change.contracts import (
    ChangeSummaryProviderResponse,
    openai_change_summary_text_format,
)
from face_of_agi.models.image_inputs import openai_image_content
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)


class OpenAIChangeSummaryProvider:
    """Final OpenAI Responses transport for transition change summaries."""

    backend = "openai"

    def __init__(
        self,
        config: Any,
        *,
        client: Any | None = None,
    ) -> None:
        if not getattr(config, "model", None):
            raise ValueError("OpenAI change summary requires an explicit model")
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
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
        """Call OpenAI and return raw change summary JSON text."""

        response = self._create_response(
            instructions_text=instructions_text,
            prompt_text=prompt_text,
            previous_image=previous_image,
            current_image=current_image,
            output_schema=output_schema,
            images=images,
            phase="complete",
            attempt=None,
        )
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
        """Ask OpenAI to repair invalid change summary JSON."""

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
        response = self._create_response(
            instructions_text=instructions_text,
            prompt_text=repair_text,
            previous_image=previous_image,
            current_image=current_image,
            output_schema=output_schema,
            images=images,
            phase="repair_complete",
            attempt=attempt,
        )
        return self._provider_response(response)

    def _create_response(
        self,
        *,
        instructions_text: str,
        prompt_text: str,
        previous_image: Any,
        current_image: Any,
        output_schema: dict[str, Any],
        images: Any | None,
        phase: str,
        attempt: int | None,
    ) -> Any:
        response = self._client.create_response(
            model=self.config.model,
            instructions=self._instructions(instructions_text),
            input_items=[
                self._input_item(
                    prompt_text,
                    previous_image=previous_image,
                    current_image=current_image,
                    images=images,
                )
            ],
            text=openai_change_summary_text_format(schema=output_schema),
        )
        self.last_request = self._client.last_request
        self._capture_request(phase=phase, response=response, attempt=attempt)
        return response

    def _provider_response(self, response: Any) -> ChangeSummaryProviderResponse:
        output_text = response_output_text(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **openai_response_metadata(response),
        }
        self.last_response_text = output_text
        self.last_response_metadata = metadata
        if output_text is None:
            raise RuntimeError(
                "OpenAI change summary response did not include output text"
            )
        return ChangeSummaryProviderResponse(
            text=output_text,
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
        capture_openai_model_input(
            self,
            call_slot="change",
            provider=self.backend,
            model=self.model,
            phase=phase,
            attempt=attempt,
            request=request,
            response=response,
        )

    def _input_item(
        self,
        prompt: str,
        *,
        previous_image: Any,
        current_image: Any,
        images: Any | None,
    ) -> dict[str, Any]:
        attached_images = tuple(images or (previous_image, current_image))
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            openai_image_content(
                attached_images,
                detail=self.config.input_image_detail,
                size=None,
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
