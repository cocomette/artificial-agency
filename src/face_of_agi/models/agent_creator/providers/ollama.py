"""Ollama provider for the agent creator roles."""

from __future__ import annotations
from typing import Any

from face_of_agi.models.agent_creator.adapter import (
    AgentCreatorAdapter,
    agent_creator_orchestrator_output_schema,
    parse_creator_orchestrator_plan_output,
)
from face_of_agi.models.agent_creator.config import OllamaAgentCreatorConfig
from face_of_agi.models.agent_creator.contracts import (
    AgentCreatorProviderResponse,
    CreatorOrchestratorRequest,
    CreatorOrchestratorResponse,
    RoleAuthorRequest,
)
from face_of_agi.models.image_inputs import ollama_image_payloads
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    assistant_json_prefill_message,
    response_usage,
    structured_json_content,
)


class OllamaAgentCreatorAdapter(AgentCreatorAdapter):
    """Agent creator backed by Ollama chat."""

    def __init__(
        self,
        config: OllamaAgentCreatorConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("Ollama agent creator requires an explicit model")
        provider = OllamaAgentCreatorProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class OllamaAgentCreatorProvider:
    """Thin Ollama translation layer for agent creator calls."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaAgentCreatorConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response: Any | None = None

    def run_orchestrator(
        self,
        request: CreatorOrchestratorRequest,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        """Return one structured creator mutation plan."""

        result = self._client.structured_chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": request.instructions},
                self._user_message(request),
                assistant_json_prefill_message(),
            ],
            response_format=agent_creator_orchestrator_output_schema(max_tool_calls),
        )
        if result.calls:
            self.last_request = result.calls[-1].request
        self.last_response = result.response
        content = structured_json_content(result.response)
        plan = parse_creator_orchestrator_plan_output(
            content,
            max_mutations=max_tool_calls,
        )
        return CreatorOrchestratorResponse(
            text=content,
            tool_call_count=len(plan.mutations),
            mutations=plan.mutations,
            metadata={
                **request.metadata,
                "backend": self.config.backend,
                "model": self.config.model,
                "usage": response_usage(self.last_response),
            },
        )

    def author_role(
        self,
        request: RoleAuthorRequest,
    ) -> AgentCreatorProviderResponse:
        """Call Ollama and return raw role-definition JSON text."""

        return self._structured_chat(request, text=request.text)

    def _user_message(
        self,
        request: CreatorOrchestratorRequest,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "user",
            "content": request.text,
        }
        if request.images:
            message["images"] = ollama_image_payloads(
                [image.image for image in request.images],
                size=self.config.input_image_size,
                resample=self.config.input_image_resample,
            )
        return message

    def repair_role(
        self,
        request: RoleAuthorRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> AgentCreatorProviderResponse:
        """Ask Ollama to repair invalid role-definition JSON."""

        repair_text = "\n\n".join(
            [
                f"Repair attempt {attempt}: previous role-author output was invalid.",
                "Validation error:\n" + validation_error,
                "Invalid output:\n" + invalid_text,
                "Original role-author input:\n" + request.text,
                "Return only corrected JSON with one role definition.",
            ]
        )
        return self._structured_chat(request, text=repair_text)

    def _structured_chat(
        self,
        request: RoleAuthorRequest,
        *,
        text: str,
    ) -> AgentCreatorProviderResponse:
        result = self._client.structured_chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": text},
                assistant_json_prefill_message(),
            ],
            response_format=request.output_schema,
        )
        if result.calls:
            self.last_request = result.calls[-1].request
        self.last_response = result.response
        content = structured_json_content(result.response)
        return AgentCreatorProviderResponse(
            text=content,
            metadata={
                **request.metadata,
                "backend": self.config.backend,
                "model": self.config.model,
                "usage": response_usage(result.response),
            },
        )
