"""OpenAI provider adapter for the agent compacter role."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.image_inputs import image_to_provider_data_url
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)
from face_of_agi.models.compacter.adapter import AgentCompacterAdapter
from face_of_agi.models.compacter.config import (
    OpenAICompacterConfig,
    with_openai_compacter_text_format,
)
from face_of_agi.models.compacter.contracts import (
    PromptCompacterProviderResponse,
    PromptCompacterRequest,
)


class OpenAICompacterAdapter(AgentCompacterAdapter):
    """Agent compacter backed by OpenAI Responses."""

    def __init__(
        self,
        config: OpenAICompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OpenAI compacter requires an explicit model")
        provider = OpenAICompacterProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OpenAICompacterProvider:
    """Thin OpenAI translation layer for the compacter role."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAICompacterConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.text = with_openai_compacter_text_format(config.text)
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def compact_context(
        self,
        request: PromptCompacterRequest,
    ) -> PromptCompacterProviderResponse:
        """Call OpenAI and return raw compacter JSON text."""

        return self._structured_response(
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
        """Ask OpenAI to repair invalid compacter JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous compacter output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original request:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request, text=repair_text)],
            text=with_openai_compacter_text_format(
                self.config.text,
                schema=request.output_schema,
                name=_schema_name(request),
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase="repair_compacter_context",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _structured_response(
        self,
        request: PromptCompacterRequest,
        *,
        phase: str,
    ) -> PromptCompacterProviderResponse:
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request)],
            text=with_openai_compacter_text_format(
                self.config.text,
                schema=request.output_schema,
                name=_schema_name(request),
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(
            phase=phase,
            request=request,
            response=response,
        )
        return self._provider_response(request, response)

    def _provider_response(
        self,
        request: PromptCompacterRequest,
        response: Any,
    ) -> PromptCompacterProviderResponse:
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
                "OpenAI compacter response did not include output text "
                f"for response {response_id!r}"
            )
        return PromptCompacterProviderResponse(
            text=output_text,
            metadata={
                **request.metadata,
                **response_metadata,
            },
        )

    def _input_item(
        self,
        request: PromptCompacterRequest,
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
            ]
            + [
                {
                    "type": "input_image",
                    "image_url": self._image_data_url(image.image),
                    "detail": self.config.input_image_detail,
                }
                for image in request.images
            ],
        }

    def _image_data_url(self, image: Any) -> str:
        return image_to_provider_data_url(
            image,
            size=None,
            resample=self.config.input_image_resample,
            mime_type=self.config.image_mime_type,
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
        capture_openai_model_input(
            self,
            call_slot="compacter",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=provider_request,
            response=response,
            attempt=attempt,
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
    "OpenAICompacterAdapter",
    "OpenAICompacterProvider",
]
