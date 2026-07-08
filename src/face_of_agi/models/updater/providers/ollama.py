"""Ollama provider for updater P."""

from __future__ import annotations

from typing import Any

from face_of_agi.models.image_inputs import frame_to_ollama_base64_png
from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)
from face_of_agi.models.updater.adapter import (
    PromptUpdaterAdapter,
)
from face_of_agi.models.updater.config import (
    OllamaUpdaterConfig,
)
from face_of_agi.models.updater.contracts import (
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
)
from face_of_agi.runtime import timing as runtime_timing


class OllamaUpdaterAdapter(PromptUpdaterAdapter):
    """Prompt updater adapter backed by Ollama chat text generation."""

    def __init__(
        self,
        config: OllamaUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama updater requires an explicit model")
        provider = OllamaUpdaterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaUpdaterProvider:
    """Thin Ollama translation layer for one prompt updater task."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaUpdaterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        """Call Ollama and return raw updater JSON text."""

        with runtime_timing.span(
            "updater.ollama_user_message",
            role=request.target.role,
            task=request.target.task,
        ):
            messages = self._messages(request)
        with runtime_timing.span(
            "updater.ollama_chat",
            role=request.target.role,
            task=request.target.task,
        ):
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
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
        """Ask Ollama to repair invalid updater JSON."""

        with runtime_timing.span(
            "updater.ollama_user_message",
            role=request.target.role,
            task=request.target.task,
            repair_attempt=attempt,
        ):
            messages = self._repair_messages(
                request,
                invalid_text=invalid_text,
                validation_error=validation_error,
                attempt=attempt,
            )
        with runtime_timing.span(
            "updater.ollama_chat",
            role=request.target.role,
            task=request.target.task,
            repair_attempt=attempt,
        ):
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                response_format=request.output_schema,
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
            assistant_json_prefill_message(),
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
            assistant_json_prefill_message(),
        ]

    def _provider_response(
        self,
        request: PromptUpdateRequest,
        response: Any,
    ) -> PromptUpdateProviderResponse:
        self.last_request = self._client.last_request
        self.last_response = response
        return PromptUpdateProviderResponse(
            target=request.target,
            text=structured_json_content(response),
            metadata={
                **request.metadata,
                "backend": self.config.backend,
                "model": self.config.model,
                "usage": response_usage(response),
            },
        )

    def _user_message(
        self,
        request: PromptUpdateRequest,
        *,
        content: str | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "user",
            "content": request.text if content is None else content,
        }
        if request.images:
            message["images"] = [
                self._image_base64(image.image) for image in request.images
            ]
        return message

    def _image_base64(self, frame: Any) -> str:
        return frame_to_ollama_base64_png(
            frame,
            frame_scale=self.config.frame_scale,
            size=self.config.input_image_size,
            resample=self.config.input_image_resample,
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
        capture_ollama_model_input(
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
    if request.target.task == "world_game":
        return (
            "Return only corrected JSON with exactly one top-level "
            "`updated_context` field. Its value must be an object containing "
            "exactly these string fields: RESET, ACTION1, ACTION2, ACTION3, "
            "ACTION4, ACTION5, ACTION6, ACTION7, NONE."
        )
    return (
        "Return only corrected JSON with exactly one top-level "
        "`updated_context` field. Its value must be a string containing the "
        "complete revised context text, not an object, array, `game` field, "
        "or `general` field."
    )
