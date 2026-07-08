"""Ollama provider for the game memory role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.image_inputs import ollama_image_payloads
from face_of_agi.models.memory.adapter import (
    GameMemoryAdapter,
    build_game_memory_repair_prompt,
)
from face_of_agi.models.memory.config import OllamaGameMemoryConfig
from face_of_agi.models.memory.contracts import (
    PromptGameMemoryProviderResponse,
    PromptGameMemoryRequest,
)
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    OllamaStructuredChatResult,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)


class OllamaGameMemoryAdapter(GameMemoryAdapter):
    """Game memory adapter backed by Ollama chat."""

    def __init__(
        self,
        config: OllamaGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama game memory requires an explicit model")
        provider = OllamaGameMemoryProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaGameMemoryProvider:
    """Thin Ollama translation layer for one memory call."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaGameMemoryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def summarize_game_memory(
        self,
        request: PromptGameMemoryRequest,
    ) -> PromptGameMemoryProviderResponse:
        """Call Ollama and return raw game memory JSON text."""

        result = self._client.structured_chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=request.output_schema,
        )
        self._capture_calls(
            result,
            phase="summarize_game_memory",
            request=request,
        )
        return self._provider_response(request, result.response)

    def repair_game_memory(
        self,
        request: PromptGameMemoryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptGameMemoryProviderResponse:
        """Ask Ollama to repair invalid game memory JSON."""

        result = self._client.structured_chat(
            model=self.config.model,
            messages=self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            ),
            response_format=request.output_schema,
        )
        self._capture_calls(
            result,
            phase="repair_game_memory",
            request=request,
            attempt=attempt,
        )
        return self._provider_response(request, result.response)

    def _provider_response(
        self,
        request: PromptGameMemoryRequest,
        response: Any,
    ) -> PromptGameMemoryProviderResponse:
        self.last_request = self._client.last_request
        self.last_response = response
        return PromptGameMemoryProviderResponse(
            text=structured_json_content(response),
            metadata={
                **request.metadata,
                "backend": self.config.backend,
                "model": self.config.model,
                "usage": response_usage(response),
            },
        )

    def _messages(self, request: PromptGameMemoryRequest) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": request.instructions},
            {
                "role": "user",
                "content": request.text,
                "images": ollama_image_payloads(
                    [item.image for item in request.images],
                    size=self.config.input_image_size,
                    resample=self.config.input_image_resample,
                    crop_box_normalized=self.config.input_image_crop_box_normalized,
                ),
            },
            assistant_json_prefill_message(),
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
            invalid_text=invalid_text,
            validation_error=validation_error,
            attempt=attempt,
            memory_max_chars=self.config.memory_max_chars,
        )
        return [
            {"role": "system", "content": request.instructions},
            {
                "role": "user",
                "content": repair_text,
            },
            assistant_json_prefill_message(),
        ]

    def _capture_calls(
        self,
        result: OllamaStructuredChatResult,
        *,
        phase: str,
        request: PromptGameMemoryRequest,
        attempt: int | None = None,
    ) -> None:
        for call in result.calls:
            call_phase = f"{phase}_thinking" if call.kind == "thinking" else phase
            capture_ollama_model_input(
                self,
                call_slot="memory",
                provider=str(self.config.backend),
                model=self.config.model,
                phase=call_phase,
                request=call.request,
                response=call.response,
                attempt=attempt,
                metadata={
                    "role": "memory",
                    "task": "game_memory",
                    "action_history_count": request.metadata.get(
                        "action_history_count"
                    ),
                },
            )
