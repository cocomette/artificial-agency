"""OpenAI provider for the agent context historizer."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.historizer.adapter import AgentContextHistorizerAdapter
from face_of_agi.models.historizer.config import (
    OpenAIHistorizerConfig,
    with_openai_agent_context_history_text_format,
)
from face_of_agi.models.historizer.contracts import (
    PromptHistorizerProviderResponse,
    PromptHistorizerRequest,
)
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)


class OpenAIHistorizerAdapter(AgentContextHistorizerAdapter):
    """Agent context historizer backed by OpenAI Responses."""

    def __init__(
        self,
        config: OpenAIHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OpenAI historizer requires an explicit model")
        provider = OpenAIHistorizerProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OpenAIHistorizerProvider:
    """Thin OpenAI translation layer for the historizer role."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAIHistorizerConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.text = with_openai_agent_context_history_text_format(config.text)
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_context_history(
        self,
        request: PromptHistorizerRequest,
    ) -> PromptHistorizerProviderResponse:
        """Call OpenAI and return raw historizer JSON text."""

        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request)],
            text=with_openai_agent_context_history_text_format(
                self.config.text,
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
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
        """Ask OpenAI to repair invalid historizer JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous historizer output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original historizer input:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request, text=repair_text)],
            text=with_openai_agent_context_history_text_format(
                self.config.text,
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="repair_context_history",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _provider_response(
        self,
        request: PromptHistorizerRequest,
        response: Any,
    ) -> PromptHistorizerProviderResponse:
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
                "OpenAI historizer response did not include output text "
                f"for response {response_id!r}"
            )
        return PromptHistorizerProviderResponse(
            text=output_text,
            metadata={
                **request.metadata,
                **response_metadata,
            },
        )

    def _input_item(
        self,
        request: PromptHistorizerRequest,
        *,
        text: str | None = None,
    ) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": request.text if text is None else text,
                }
            ],
        }

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
        capture_openai_model_input(
            self,
            call_slot="historizer",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=provider_request,
            response=response,
            attempt=attempt,
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
