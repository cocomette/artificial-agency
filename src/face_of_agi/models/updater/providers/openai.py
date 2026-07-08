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
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="update_prompt",
            request=request,
            response=response,
        )
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

    def _input_item(self, request: PromptUpdateRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": request.text,
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
            frame_scale=self.config.frame_scale,
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
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None or request.target.task == "general":
            return
        capture_openai_model_input(
            self,
            call_slot=f"updater_{request.target.role}",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=provider_request,
            response=response,
            metadata={
                "role": request.target.role,
                "segment": request.target.segment,
                "task": request.target.task,
            },
        )
