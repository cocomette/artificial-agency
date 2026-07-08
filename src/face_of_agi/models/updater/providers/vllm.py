"""vLLM provider for updater P."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.image_inputs import frame_to_provider_data_url
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)
from face_of_agi.models.updater.adapter import PromptUpdaterAdapter
from face_of_agi.models.updater.config import VLLMUpdaterConfig
from face_of_agi.models.updater.contracts import (
    AGENT_GAME_CONTEXT_MAX_CHARS,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
)


class VLLMUpdaterAdapter(PromptUpdaterAdapter):
    """Prompt updater adapter backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM updater requires an explicit model")
        provider = VLLMUpdaterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMUpdaterProvider:
    """Thin vLLM translation layer for one prompt updater task."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        """Call vLLM and return raw updater JSON text."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=json_schema_response_format(
                name="updater_context_update",
                schema=request.output_schema,
            ),
        )
        self._capture_request(
            phase="update_prompt",
            request=request,
            response=response,
        )
        return self._provider_response(request, response)

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        """Ask vLLM to repair invalid updater JSON."""

        response = self._client.chat(
            model=self.config.model,
            messages=self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            ),
            response_format=json_schema_response_format(
                name="updater_context_update",
                schema=request.output_schema,
            ),
        )
        self._capture_request(
            phase="repair_prompt",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _messages(self, request: PromptUpdateRequest) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request),
        ]

    def _repair_messages(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous updater output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original updater input:\n" + request.text,
                _repair_output_instruction(request),
            ]
        )
        return [
            {
                "role": "system",
                "content": request.instructions,
            },
            self._user_message(request, content=repair_text),
        ]

    def _provider_response(
        self,
        request: PromptUpdateRequest,
        response: Any,
    ) -> PromptUpdateProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_optional_content(response) or ""
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptUpdateProviderResponse(
            target=request.target,
            text=text,
            metadata={
                **request.metadata,
                **self.last_response_metadata,
            },
        )

    def _user_message(
        self,
        request: PromptUpdateRequest,
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
                        "url": self._image_data_url(image.image),
                        "detail": self.config.input_image_detail,
                    },
                }
            )
        return {"role": "user", "content": message_content}

    def _image_data_url(self, frame: Any) -> str:
        return frame_to_provider_data_url(
            frame,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
            mime_type=self.config.image_mime_type,
        )

    def _capture_request(
        self,
        *,
        phase: str,
        request: PromptUpdateRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None or request.target.task == "general":
            return
        capture_vllm_model_input(
            self,
            call_slot=f"updater_{request.target.role}",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
            metadata={
                "role": request.target.role,
                "segment": request.target.segment,
                "task": request.target.task,
            },
        )


def _repair_output_instruction(request: PromptUpdateRequest) -> str:
    if request.target.task in {"agent_probing", "agent_policy"}:
        return (
            "Return only corrected JSON with exactly the summary field "
            "required by this updater role and one top-level `next_actions` "
            "field. The full serialized context must be no more "
            f"than {AGENT_GAME_CONTEXT_MAX_CHARS} characters. Use enough "
            "detail to preserve useful current context and delete stale or "
            "duplicate details. `next_actions` must contain exactly the number "
            "of items required by the schema, and every item must validate "
            "against the "
            "allowed-action schema."
        )
    return "Return only corrected JSON that validates against the schema."
