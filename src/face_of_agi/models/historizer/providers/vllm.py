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

        return self._structured_chat(
            request,
            phase="summarize_context_history",
        )

    def _structured_chat(
        self,
        request: PromptHistorizerRequest,
        *,
        phase: str,
    ) -> PromptHistorizerProviderResponse:
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=json_schema_response_format(
                name=_schema_name(request),
                schema=request.output_schema,
            ),
        )
        self._capture_request(
            phase=phase,
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
                name=_schema_name(request),
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
            self._user_message(request),
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
                (
                    f"Repair attempt {attempt}: the previous "
                    "historizer output was invalid."
                ),
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
            self._user_message(request, content=repair_text),
        ]

    def _user_message(
        self,
        request: PromptHistorizerRequest,
        *,
        content: str | None = None,
    ) -> dict[str, Any]:
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": request.text if content is None else content,
            }
        ]
        return {"role": "user", "content": message_content}

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
            call_slot=_call_slot(request),
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
            metadata={
                "role": _call_slot(request),
                "task": request.metadata.get("task", "agent_context_history"),
            },
        )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly these top-level fields: "
        "`probing_evolution`, `policy_evolution`, and `updater_mode`. "
        "`probing_evolution` and `policy_evolution` must be strings; "
        "`updater_mode` must be `probing` or `policy`."
    )


def _schema_name(request: PromptHistorizerRequest) -> str:
    return str(request.metadata.get("schema_name") or "agent_context_history")


def _call_slot(request: PromptHistorizerRequest) -> str:
    return "historizer"
