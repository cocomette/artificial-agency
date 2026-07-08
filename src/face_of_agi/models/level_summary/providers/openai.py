"""OpenAI provider for the per-level solution summarizer."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.level_summary.adapter import LevelSolutionSummarizerAdapter
from face_of_agi.models.level_summary.config import (
    OpenAILevelSummaryConfig,
    with_openai_level_solution_summary_text_format,
)
from face_of_agi.models.level_summary.contracts import (
    PromptLevelSummaryProviderResponse,
    PromptLevelSummaryRequest,
)
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    openai_response_metadata,
    response_output_text,
)


class OpenAILevelSummaryAdapter(LevelSolutionSummarizerAdapter):
    """Level summarizer backed by OpenAI Responses."""

    def __init__(
        self,
        config: OpenAILevelSummaryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("OpenAI level summary requires an explicit model")
        provider = OpenAILevelSummaryProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OpenAILevelSummaryProvider:
    """Thin OpenAI translation layer for the level-summary role."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAILevelSummaryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.text = with_openai_level_solution_summary_text_format(config.text)
        self.config = config
        self.model = config.model
        self._client = OpenAIResponsesClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_level_solution(
        self,
        request: PromptLevelSummaryRequest,
    ) -> PromptLevelSummaryProviderResponse:
        """Call OpenAI and return raw level-summary JSON text."""

        return self._structured_response(
            request,
            phase="summarize_level_solution",
        )

    def repair_level_solution(
        self,
        request: PromptLevelSummaryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptLevelSummaryProviderResponse:
        """Ask OpenAI to repair invalid level-summary JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous level summary output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original level-summary input:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        return self._repair_response(
            request,
            repair_text=repair_text,
            phase="repair_level_solution",
            attempt=attempt,
        )

    def _structured_response(
        self,
        request: PromptLevelSummaryRequest,
        *,
        phase: str,
    ) -> PromptLevelSummaryProviderResponse:
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request)],
            text=with_openai_level_solution_summary_text_format(
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

    def _repair_response(
        self,
        request: PromptLevelSummaryRequest,
        *,
        repair_text: str,
        phase: str,
        attempt: int,
    ) -> PromptLevelSummaryProviderResponse:
        response = self._client.create_response(
            model=self.config.model,
            instructions=request.instructions,
            input_items=[self._input_item(request, text=repair_text)],
            text=with_openai_level_solution_summary_text_format(
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
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _provider_response(
        self,
        request: PromptLevelSummaryRequest,
        response: Any,
    ) -> PromptLevelSummaryProviderResponse:
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
                "OpenAI level summary response did not include output text "
                f"for response {response_id!r}"
            )
        return PromptLevelSummaryProviderResponse(
            text=output_text,
            metadata={
                **request.metadata,
                **response_metadata,
            },
        )

    def _input_item(
        self,
        request: PromptLevelSummaryRequest,
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
        request: PromptLevelSummaryRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_openai_model_input(
            self,
            call_slot="level_summary",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            request=provider_request,
            response=response,
            attempt=attempt,
            metadata={
                "role": "level_summary",
                "task": request.metadata.get("task", "level_solution_summary"),
            },
        )


def _repair_output_instruction() -> str:
    return (
        "Return only corrected JSON with exactly one top-level string field: "
        "`solution_method`."
    )


def _schema_name(request: PromptLevelSummaryRequest) -> str:
    return str(request.metadata.get("schema_name") or "level_solution_summary")


__all__ = [
    "OpenAILevelSummaryAdapter",
    "OpenAILevelSummaryProvider",
]
