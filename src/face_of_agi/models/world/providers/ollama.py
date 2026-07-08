"""Ollama provider adapter for the agent world-model role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.image_inputs import image_to_ollama_base64_png
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    OllamaStructuredChatResult,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)
from face_of_agi.models.world.adapter import AgentWorldModelAdapter
from face_of_agi.models.world.config import OllamaWorldModelConfig
from face_of_agi.models.world.contracts import (
    PromptWorldProviderResponse,
    PromptWorldRequest,
)
from face_of_agi.runtime import timing as runtime_timing


class OllamaWorldModelAdapter(AgentWorldModelAdapter):
    """Agent world model backed by Ollama chat."""

    def __init__(
        self,
        config: OllamaWorldModelConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama world model requires an explicit model")
        provider = OllamaWorldModelProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaWorldModelProvider:
    """Thin Ollama translation layer for the world-model role."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaWorldModelConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def summarize_world_model(
        self,
        request: PromptWorldRequest,
    ) -> PromptWorldProviderResponse:
        """Call Ollama and return raw world-model JSON text."""

        return self._structured_chat(
            request,
            phase="summarize_world_model",
        )

    def repair_world_model(
        self,
        request: PromptWorldRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptWorldProviderResponse:
        """Ask Ollama to repair invalid world-model JSON."""

        with runtime_timing.span(
            "world_model.ollama_user_message",
            repair_attempt=attempt,
        ):
            messages = self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            )
        with runtime_timing.span(
            "world_model.ollama_chat",
            repair_attempt=attempt,
        ):
            result = self._client.structured_chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
            )
            self._capture_calls(
                result,
                phase="repair_world_model",
                request=request,
                attempt=attempt,
            )
        return self._provider_response(request, result.response)

    def _structured_chat(
        self,
        request: PromptWorldRequest,
        *,
        phase: str,
    ) -> PromptWorldProviderResponse:
        with runtime_timing.span("world_model.ollama_user_message"):
            messages = self._messages(request)
        with runtime_timing.span("world_model.ollama_chat"):
            result = self._client.structured_chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
            )
            self._capture_calls(
                result,
                phase=phase,
                request=request,
            )
        return self._provider_response(request, result.response)

    def _messages(self, request: PromptWorldRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request),
            assistant_json_prefill_message(),
        ]

    def _repair_messages(
        self,
        request: PromptWorldRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous world_model output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original world_model input:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request, content=repair_text),
            assistant_json_prefill_message(),
        ]

    def _user_message(
        self,
        request: PromptWorldRequest,
        *,
        content: str | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "user",
            "content": request.text if content is None else content,
        }
        if request.images:
            message["images"] = [
                image_to_ollama_base64_png(
                    image.image,
                    size=None,
                    resample=self.config.input_image_resample,
                )
                for image in request.images
            ]
        return message

    def _provider_response(
        self,
        request: PromptWorldRequest,
        response: Any,
    ) -> PromptWorldProviderResponse:
        self.last_request = self._client.last_request
        self.last_response = response
        return PromptWorldProviderResponse(
            text=structured_json_content(response),
            metadata={
                **request.metadata,
                "backend": self.config.backend,
                "model": self.config.model,
                "usage": response_usage(response),
            },
        )

    def _capture_calls(
        self,
        result: OllamaStructuredChatResult,
        *,
        phase: str,
        request: PromptWorldRequest,
        attempt: int | None = None,
    ) -> None:
        for call in result.calls:
            call_phase = f"{phase}_thinking" if call.kind == "thinking" else phase
            capture_ollama_model_input(
                self,
                call_slot="world_model",
                provider=str(self.config.backend),
                model=self.config.model,
                phase=call_phase,
                attempt=attempt,
                request=call.request,
                response=call.response,
                metadata={
                    "role": "world_model",
                    "task": request.metadata.get("task", "agent_world_model"),
                },
            )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly these top-level fields: "
        "`world_description`, `special_events`, and `action_effects`. "
        "`world_description` and `special_events` must be strings and "
        "`action_effects` must validate against the schema."
    )


__all__ = [
    "OllamaWorldModelAdapter",
    "OllamaWorldModelProvider",
]
