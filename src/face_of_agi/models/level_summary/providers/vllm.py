"""vLLM provider for the per-level solution summarizer."""

from __future__ import annotations

from typing import Any

from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.level_summary.adapter import LevelSolutionSummarizerAdapter
from face_of_agi.models.level_summary.config import VLLMLevelSummaryConfig
from face_of_agi.models.level_summary.contracts import (
    PromptLevelSummaryProviderResponse,
    PromptLevelSummaryRequest,
)
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)


class VLLMLevelSummaryAdapter(LevelSolutionSummarizerAdapter):
    """Level summarizer backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMLevelSummaryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM level summary requires an explicit model")
        provider = VLLMLevelSummaryProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMLevelSummaryProvider:
    """Thin vLLM translation layer for the level-summary role."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMLevelSummaryConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def summarize_level_solution(
        self,
        request: PromptLevelSummaryRequest,
    ) -> PromptLevelSummaryProviderResponse:
        """Call vLLM and return raw level-summary JSON text."""

        return self._structured_chat(request, phase="summarize_level_solution")

    def repair_level_solution(
        self,
        request: PromptLevelSummaryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptLevelSummaryProviderResponse:
        """Ask vLLM to repair invalid level-summary JSON."""

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
            phase="repair_level_solution",
            request=request,
            response=response,
            attempt=attempt,
        )
        return self._provider_response(request, response)

    def _structured_chat(
        self,
        request: PromptLevelSummaryRequest,
        *,
        phase: str,
    ) -> PromptLevelSummaryProviderResponse:
        response = self._client.chat(
            model=self.config.model,
            messages=self._messages(request),
            response_format=json_schema_response_format(
                name=_schema_name(request),
                schema=request.output_schema,
            ),
        )
        self._capture_request(phase=phase, request=request, response=response)
        return self._provider_response(request, response)

    def _messages(self, request: PromptLevelSummaryRequest) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": request.text},
        ]

    def _repair_messages(
        self,
        request: PromptLevelSummaryRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> list[dict[str, Any]]:
        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: the previous level summary output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original level-summary input:\n" + request.text,
                _repair_output_instruction(),
            ]
        )
        return [
            {"role": "system", "content": request.instructions},
            {"role": "user", "content": repair_text},
        ]

    def _provider_response(
        self,
        request: PromptLevelSummaryRequest,
        response: Any,
    ) -> PromptLevelSummaryProviderResponse:
        response_metadata = chat_response_metadata(response)
        text = chat_message_optional_content(response) or ""
        self.last_request = self._client.last_request
        self.last_response_text = text
        self.last_response_metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **response_metadata,
        }
        return PromptLevelSummaryProviderResponse(
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
        request: PromptLevelSummaryRequest,
        response: Any | None,
        attempt: int | None = None,
    ) -> None:
        provider_request = self._client.last_request
        if provider_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="level_summary",
            provider=str(self.config.backend),
            model=self.config.model,
            phase=phase,
            attempt=attempt,
            request=provider_request,
            response=response,
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
    "VLLMLevelSummaryAdapter",
    "VLLMLevelSummaryProvider",
]
