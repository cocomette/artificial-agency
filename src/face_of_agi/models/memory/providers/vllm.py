"""vLLM provider for the game memory role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.memory.adapter import (
    GameMemoryAdapter,
    build_game_memory_repair_prompt,
)
from face_of_agi.models.memory.config import VLLMGameMemoryConfig
from face_of_agi.models.memory.contracts import (
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
)
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.structured_output import clipped_invalid_output_preview


class VLLMGameMemoryAdapter(GameMemoryAdapter):
    """Game memory adapter backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM game memory requires an explicit model")
        provider = VLLMGameMemoryProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMGameMemoryProvider:
    """Thin vLLM translation layer for one memory call."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_game_memory(
        self,
        request: PromptGameMemoryRequest,
    ) -> PromptGameMemoryProviderResponse:
        """Call vLLM and return raw game memory JSON text."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=json_schema_response_format(
                name="game_memory",
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="summarize_game_memory",
            request=request,
            response=response,
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
        """Ask vLLM to repair invalid game memory JSON."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            ),
            response_format=json_schema_response_format(
                name="game_memory",
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="repair_game_memory",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _messages(self, request: PromptGameMemoryRequest) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": request.instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.text},
                    *vllm_image_content(
                        [item.image for item in request.images],
                        detail=self.config.input_image_detail,
                        size=self.config.input_image_size,
                        resample=self.config.input_image_resample,
                        mime_type=self.config.image_mime_type,
                        crop_box_normalized=self.config.input_image_crop_box_normalized,
                    ),
                ],
            },
        ]

    def _repair_messages(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = build_game_memory_repair_prompt(
            invalid_text=clipped_invalid_output_preview(
                invalid_text,
                max_chars=self.config.repair_invalid_output_preview_chars,
            ),
            validation_error=validation_error,
            attempt=attempt,
            memory_max_chars=self.config.memory_max_chars,
        )
        return [
            {"role": "system", "content": request.instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": repair_text},
                ],
            },
        ]

    def _provider_response(
        self,
        request: PromptGameMemoryRequest,
        response: Any,
    ) -> PromptGameMemoryProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_content(response)
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptGameMemoryProviderResponse(
            text=text,
            metadata={
                **request.metadata,
                **self.last_response_metadata,
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
        capture_vllm_model_input(
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
