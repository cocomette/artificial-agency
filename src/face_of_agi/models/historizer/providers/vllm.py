"""vLLM provider for the agent context historizer."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.historizer.adapter import AgentContextHistorizerAdapter
from face_of_agi.models.historizer.config import VLLMHistorizerConfig
from face_of_agi.models.historizer.contracts import (
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
)
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)


class VLLMHistorizerAdapter(AgentContextHistorizerAdapter):
    """Agent context historizer backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM historizer requires an explicit model")
        provider = VLLMHistorizerProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMHistorizerProvider:
    """Thin vLLM translation layer for the historizer role."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_context_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        """Call vLLM and return raw historizer JSON text."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=json_schema_response_format(
                name="agent_context_history",
                schema=request.output_schema,
            ),
        )
        self._capture_request(
            phase="summarize_context_history",
            request=request,
            response=response,
        )
        return self._provider_response(request, response)

    def repair_context_history(
        self,
        request: PromptHistorizerRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptHistorizerProviderResponse:
        """Ask vLLM to repair invalid historizer JSON."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            ),
            response_format=json_schema_response_format(
                name="agent_context_history",
                schema=request.output_schema,
            ),
        )
        self._capture_request(
            phase="repair_context_history",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _messages(self, request: PromptHistorizerRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            {
                "role": "user",
                "content": request.text,
            },
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
            {
                "role": "system",
                "content": request.instructions,
            },
            {
                "role": "user",
                "content": repair_text,
            },
        ]

    def _provider_response(
        self,
        request: PromptHistorizerRequest,
        response: Any,
    ) -> PromptHistorizerProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_optional_content(response) or ""
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptHistorizerProviderResponse(
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
        request: PromptHistorizerRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="historizer",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
            metadata={
                "role": "historizer",
                "task": "agent_context_history",
            },
        )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly one top-level "
        "`field_evolution` field. Its value must be an object containing "
        "exactly these string fields: goals, game_mechanics, policy, history, "
        "extras. Use enough detail to capture field evolution while staying "
        "trend-focused."
    )
