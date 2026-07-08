"""Ollama provider for the historizer."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.historizer.adapter import HistorizerAdapter
from face_of_agi.models.historizer.config import OllamaHistorizerConfig
from face_of_agi.models.historizer.contracts import (
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
)
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    OllamaStructuredChatResult,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)
from face_of_agi.runtime import timing as runtime_timing


class OllamaHistorizerAdapter(HistorizerAdapter):
    """Historizer backed by Ollama chat."""

    def __init__(
        self,
        config: OllamaHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama historizer requires an explicit model")
        provider = OllamaHistorizerProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaHistorizerProvider:
    """Thin Ollama translation layer for the historizer role."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def summarize_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        """Call Ollama and return raw historizer JSON text."""

        return self._structured_chat(request, phase="summarize_history")

    def repair_history(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptHistorizerProviderResponse:
        """Ask Ollama to repair invalid historizer JSON."""

        with runtime_timing.span("historizer.ollama_user_message", repair_attempt=attempt):
            messages = self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            )
        with runtime_timing.span("historizer.ollama_chat", repair_attempt=attempt):
            result = self._client.structured_chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
            )
            self._capture_calls(
                result,
                phase="repair_history",
                request=request,
                attempt=attempt,
            )
        return self._provider_response(request, result.response)

    def _structured_chat(
        self,
        request: PromptHistorizerRequest,
        *,
        phase: str,
    ) -> PromptHistorizerProviderResponse:
        with runtime_timing.span("historizer.ollama_user_message"):
            messages = self._messages(request)
        with runtime_timing.span("historizer.ollama_chat"):
            result = self._client.structured_chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
            )
            self._capture_calls(result, phase=phase, request=request)
        return self._provider_response(request, result.response)

    def _messages(self, request: PromptHistorizerRequest) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": request.text},
            assistant_json_prefill_message(),
        ]

    def _repair_messages(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous historizer output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original historizer input:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        return [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": repair_text},
            assistant_json_prefill_message(),
        ]

    def _provider_response(
        self,
        request: PromptHistorizerRequest,
        response: Any,
    ) -> PromptHistorizerProviderResponse:
        self.last_request = self._client.last_request
        self.last_response = response
        return PromptHistorizerProviderResponse(
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
        request: PromptHistorizerRequest,
        attempt: int | None = None,
    ) -> None:
        for call in result.calls:
            call_phase = f"{phase}_thinking" if call.kind == "thinking" else phase
            capture_ollama_model_input(
                self,
                call_slot="historizer",
                provider=str(self.config.backend),
                model=self.config.model,
                phase=call_phase,
                attempt=attempt,
                request=call.request,
                response=call.response,
                metadata={
                    "role": "historizer",
                    "task": request.metadata.get("task", "history_summary"),
                },
            )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly two top-level string fields: "
        "`action_history_summary` and `strategy_history_summary`."
    )


__all__ = [
    "OllamaHistorizerAdapter",
    "OllamaHistorizerProvider",
]
