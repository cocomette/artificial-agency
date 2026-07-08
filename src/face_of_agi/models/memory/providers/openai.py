"""OpenAI provider for the game memory role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.image_inputs import openai_image_content
from face_of_agi.models.memory.adapter import (
    GameMemoryAdapter,
    build_game_memory_repair_prompt,
)
from face_of_agi.models.memory.config import (
    OpenAIGameMemoryConfig,
    with_openai_game_memory_text_format,
)
from face_of_agi.models.memory.contracts import (
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
)
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)


class OpenAIGameMemoryAdapter(GameMemoryAdapter):
    """Game memory adapter backed by OpenAI Responses."""

    def __init__(
        self,
        config: OpenAIGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OpenAI game memory requires an explicit model")
        provider = OpenAIGameMemoryProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OpenAIGameMemoryProvider:
    """Thin OpenAI translation layer for one memory call."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAIGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.text = with_openai_game_memory_text_format(config.text)
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_game_memory(
        self,
        request: PromptGameMemoryRequest,
    ) -> PromptGameMemoryProviderResponse:
        """Call OpenAI and return raw game memory JSON text."""

        response = self._create_response(
            request,
            phase="summarize_game_memory",
            attempt=None,
        )
        return self._provider_response(request, response)

    def repair_game_memory(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptGameMemoryProviderResponse:
        """Ask OpenAI to repair invalid game memory JSON."""

        repair_text = build_game_memory_repair_prompt(
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
            memory_max_chars=self.config.memory_max_chars,
        )
        response = self._create_response(
            request,
            phase="repair_game_memory",
            attempt=attempt,
            text=repair_text,
            include_images=False,
        )
        return self._provider_response(request, response)

    def _create_response(
        self,
        request: PromptGameMemoryRequest,
        *,
        phase: str,
        attempt: int | None,
        text: str | None = None,
        include_images: bool = True,
    ) -> Any:
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[
                self._input_item(
                    request,
                    text=text,
                    include_images=include_images,
                )
            ],
            text=with_openai_game_memory_text_format(
                self.config.text,
                schema=request.output_schema,
            ),
            include_max_tool_calls=False,
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase=phase,
            request=request,
            response=response,
            attempt=attempt,
        )
        return response

    def _input_item(
        self,
        request: PromptGameMemoryRequest,
        *,
        text: str | None = None,
        include_images: bool = True,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": request.text if text is None else text}
        ]
        if include_images:
            content.extend(
                openai_image_content(
                    [item.image for item in request.images],
                    detail=self.config.input_image_detail,
                    size=self.config.input_image_size,
                    resample=self.config.input_image_resample,
                    mime_type=self.config.image_mime_type,
                    crop_box_normalized=self.config.input_image_crop_box_normalized,
                )
            )
        return {"role": "user", "content": content}

    def _provider_response(
        self,
        request: PromptGameMemoryRequest,
        response: Any,
    ) -> PromptGameMemoryProviderResponse:
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
                "OpenAI game memory response did not include output text "
                f"for response {response_id!r}"
            )
        return PromptGameMemoryProviderResponse(
            text=output_text,
            metadata={
                **request.metadata,
                **response_metadata,
            },
        )

    def _capture_request(
        self,
        *,
        phase: str,
        request: PromptGameMemoryRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        if self.last_request is None:
            return
        capture_openai_model_input(
            self,
            call_slot="memory",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=self.last_request,
            response=response,
            attempt=attempt,
            metadata={
                "role": "memory",
                "task": "game_memory",
                "action_history_count": request.metadata.get("action_history_count"),
            },
        )
