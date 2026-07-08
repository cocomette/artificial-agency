"""vLLM provider adapter for the agent compacter role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.image_inputs import image_to_provider_data_url
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.compacter.adapter import AgentCompacterAdapter
from face_of_agi.models.compacter.config import VLLMCompacterConfig
from face_of_agi.models.compacter.contracts import (
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
)


class VLLMCompacterAdapter(AgentCompacterAdapter):
    """Agent compacter backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMCompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM compacter requires an explicit model")
        provider = VLLMCompacterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMCompacterProvider:
    """Thin vLLM translation layer for the compacter role."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMCompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        """Call vLLM and return raw compacter JSON text."""

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
        """Ask vLLM to repair invalid compacter JSON."""

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
            phase="repair_compacter_context",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _structured_chat(
        self,
        request: PromptCompacterRequest,
        *,
        phase: str,
    ) -> PromptCompacterProviderResponse:
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

    def _messages(self, request: PromptCompacterRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request),
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
        ]

    def _user_message(
        self,
        request: PromptCompacterRequest,
        *,
        content: str | None = None,
    ) -> dict[str, Any]:
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": request.text if content is None else content,
            }
        ]
        for image in request.images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_provider_data_url(
                            image.image,
                            size=None,
                            resample=self.config.input_image_resample,
                            mime_type=self.config.image_mime_type,
                        ),
                        "detail": self.config.input_image_detail,
                    },
                }
            )
        return {"role": "user", "content": message_content}

    def _provider_response(
        self,
        request: PromptCompacterRequest,
        response: Any,
    ) -> PromptCompacterProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_optional_content(response) or ""
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptCompacterProviderResponse(
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
        request: PromptCompacterRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="compacter",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
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


def _schema_name(request: PromptCompacterRequest) -> str:
    return str(request.metadata.get("schema_name") or "agent_compacter")


__all__ = [
    "VLLMCompacterAdapter",
    "VLLMCompacterProvider",
]
