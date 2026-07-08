"""Ollama provider adapter for the agent compacter role."""

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
from face_of_agi.models.compacter.adapter import AgentCompacterAdapter
from face_of_agi.models.compacter.config import OllamaCompacterConfig
from face_of_agi.models.compacter.contracts import (
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
)
from face_of_agi.runtime import timing as runtime_timing


class OllamaCompacterAdapter(AgentCompacterAdapter):
    """Agent compacter backed by Ollama chat."""

    def __init__(
        self,
        config: OllamaCompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama compacter requires an explicit model")
        provider = OllamaCompacterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaCompacterProvider:
    """Thin Ollama translation layer for the compacter role."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaCompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        """Call Ollama and return raw compacter JSON text."""

        return self._structured_chat(
            request,
            phase="compact_context",
        )

    def repair_compacter_context(
        self,
        request: PromptCompacterRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptCompacterProviderResponse:
        """Ask Ollama to repair invalid compacter JSON."""

        with runtime_timing.span(
            "compacter.ollama_user_message",
            repair_attempt=attempt,
        ):
            messages = self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            )
        with runtime_timing.span(
            "compacter.ollama_chat",
            repair_attempt=attempt,
        ):
            result = self._client.structured_chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
            )
            self._capture_calls(
                result,
                phase="repair_compacter_context",
                request=request,
                attempt=attempt,
            )
        return self._provider_response(request, result.response)

    def _structured_chat(
        self,
        request: PromptCompacterRequest,
        *,
        phase: str,
    ) -> PromptCompacterProviderResponse:
        with runtime_timing.span("compacter.ollama_user_message"):
            messages = self._messages(request)
        with runtime_timing.span("compacter.ollama_chat"):
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

    def _messages(self, request: PromptCompacterRequest) -> list[dict[str, Any]]:
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
        request: PromptCompacterRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous compacter output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original request:\n" + request.text,
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
        request: PromptCompacterRequest,
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
        request: PromptCompacterRequest,
        response: Any,
    ) -> PromptCompacterProviderResponse:
        self.last_request = self._client.last_request
        self.last_response = response
        return PromptCompacterProviderResponse(
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
        request: PromptCompacterRequest,
        attempt: int | None = None,
    ) -> None:
        for call in result.calls:
            call_phase = f"{phase}_thinking" if call.kind == "thinking" else phase
            capture_ollama_model_input(
                self,
                call_slot="compacter",
                provider=str(self.config.backend),
                model=self.config.model,
                phase=call_phase,
                attempt=attempt,
                request=call.request,
                response=call.response,
                metadata={
                    "role": "compacter",
                    "task": request.metadata.get("task", "agent_compacter"),
                },
            )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly these top-level fields: "
        "`world_description`, `special_events`, `action_effects`, "
        "`previous_actions_summary`, and `previous_strategy_summary`. "
        "`world_description`, `special_events`, `previous_actions_summary`, "
        "and `previous_strategy_summary` must be strings and `action_effects` "
        "must validate against the schema."
    )


__all__ = [
    "OllamaCompacterAdapter",
    "OllamaCompacterProvider",
]
