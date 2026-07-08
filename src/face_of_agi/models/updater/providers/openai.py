"""OpenAI provider for updater P."""

from __future__ import annotations

from typing import Any

from face_of_agi.models.image_inputs import frame_to_provider_data_url
from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)
from face_of_agi.models.updater.adapter import (
    PromptUpdaterAdapter,
)
from face_of_agi.models.updater.config import (
    OpenAIUpdaterConfig,
    with_openai_updated_context_text_format,
)
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_MAX_CHARS,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
)


class OpenAIUpdaterAdapter(PromptUpdaterAdapter):
    """Prompt updater adapter backed by OpenAI Responses text generation."""

    def __init__(
        self,
        config: OpenAIUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OpenAI updater requires an explicit model")
        provider = OpenAIUpdaterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OpenAIUpdaterProvider:
    """Thin OpenAI translation layer for one prompt updater task."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAIUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.text = with_openai_updated_context_text_format(config.text)
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        """Call OpenAI and return raw updater JSON text."""

        self.last_request = None
        self.last_response_text = None
        self.last_response_metadata = None
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request)],
            text=with_openai_updated_context_text_format(
                self.config.text,
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="update_prompt",
            request=request,
            response=response,
        )
        return self._provider_response(request, response)

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        """Ask OpenAI to repair invalid updater JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous updater output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original updater input:\n" + request.text,
                _repair_output_instruction(request),
            ]
        )
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request, text=repair_text)],
            text=with_openai_updated_context_text_format(
                self.config.text,
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="repair_prompt",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _provider_response(
        self,
        request: PromptUpdateRequest,
        response: Any,
    ) -> PromptUpdateProviderResponse:
        output_text = response_output_text(response)
        response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **openai_response_metadata(response),
        }
        self.last_response_text = output_text
        self.last_response_metadata = response_metadata
        if output_text is None:
            response_id = response_metadata.get("response_id")
            raise RuntimeError(
                "OpenAI updater response did not include output text "
                f"for response {response_id!r}"
            )
        return PromptUpdateProviderResponse(
            target=request.target,
            text=output_text,
            metadata={
                **request.metadata,
                **response_metadata,
            },
        )

    def _input_item(
        self,
        request: PromptUpdateRequest,
        *,
        text: str | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": request.text if text is None else text,
            }
        ]
        content.extend(
            {
                "type": "input_image",
                "image_url": self._image_data_url(image.image),
                "detail": self.config.input_image_detail,
            }
            for image in request.images
        )
        return {"role": "user", "content": content}

    def _image_data_url(self, frame: Any) -> str:
        return frame_to_provider_data_url(
            frame,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
            mime_type=self.config.image_mime_type,
        )

    def _capture_request(
        self,
        *,
        phase: str,
        request: PromptUpdateRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_openai_model_input(
            self,
            call_slot=f"updater_{request.target.role}",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=provider_request,
            response=response,
            attempt=attempt,
            metadata={
                "role": request.target.role,
                "segment": request.target.segment,
                "task": request.target.task,
            },
        )


def _repair_output_instruction(request: PromptUpdateRequest) -> str:
    if request.target.task == "agent":
        return (
            "Return only corrected JSON with exactly `current_strategy` and "
            "one top-level `next_actions` field. The "
            "full serialized context must be no more "
            f"than {AGENT_GAME_CONTEXT_MAX_CHARS} characters. Use enough "
            "detail to preserve useful current context and delete stale or "
            "duplicate details. `next_actions` must contain exactly the number "
            "of items required by the schema, and every item must validate "
            "against the "
            "allowed-action schema."
        )
    return (
        "Return only corrected JSON with exactly one top-level "
        "`updated_context` field. Its value must be a string containing the "
        "complete revised context text, not an object, array, `game` field, "
        "or `general` field."
    )
