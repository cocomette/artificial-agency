"""vLLM provider adapter for the agent world-model role."""

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
from face_of_agi.models.world.adapter import AgentWorldModelAdapter
from face_of_agi.models.world.config import VLLMWorldModelConfig
from face_of_agi.models.world.contracts import (
    PromptWorldProviderResponse,
    PromptWorldRequest,
)


class VLLMWorldModelAdapter(AgentWorldModelAdapter):
    """Agent world model backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMWorldModelConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM world model requires an explicit model")
        provider = VLLMWorldModelProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMWorldModelProvider:
    """Thin vLLM translation layer for the world-model role."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMWorldModelConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_world_model(
        self,
        request: PromptWorldRequest,
    ) -> PromptWorldProviderResponse:
        """Call vLLM and return raw world-model JSON text."""

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
        """Ask vLLM to repair invalid world-model JSON."""

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
            phase="repair_world_model",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _structured_chat(
        self,
        request: PromptWorldRequest,
        *,
        phase: str,
    ) -> PromptWorldProviderResponse:
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

    def _messages(self, request: PromptWorldRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request),
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
        ]

    def _user_message(
        self,
        request: PromptWorldRequest,
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
        request: PromptWorldRequest,
        response: Any,
    ) -> PromptWorldProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_optional_content(response) or ""
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptWorldProviderResponse(
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
        request: PromptWorldRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="world_model",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
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


def _schema_name(request: PromptWorldRequest) -> str:
    return str(request.metadata.get("schema_name") or "agent_world_model")


__all__ = [
    "VLLMWorldModelAdapter",
    "VLLMWorldModelProvider",
]
