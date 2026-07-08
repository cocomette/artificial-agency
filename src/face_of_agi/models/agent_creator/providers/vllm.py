"""vLLM provider for the agent creator roles."""

from __future__ import annotations
from typing import Any

from face_of_agi.models.agent_creator.adapter import (
    AgentCreatorAdapter,
    agent_creator_orchestrator_output_schema,
    parse_creator_orchestrator_plan_output,
)
from face_of_agi.models.agent_creator.config import VLLMAgentCreatorConfig
from face_of_agi.models.agent_creator.contracts import (
    AgentCreatorProviderResponse,
    CreatorOrchestratorRequest,
    CreatorOrchestratorResponse,
    RoleAuthorRequest,
)
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message_optional_content,
    chat_response_metadata,
    json_schema_response_format,
)


class VLLMAgentCreatorAdapter(AgentCreatorAdapter):
    """Agent creator backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMAgentCreatorConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM agent creator requires an explicit model")
        provider = VLLMAgentCreatorProvider(config, client=client)
        super().__init__(provider=provider, config=config)


class VLLMAgentCreatorProvider:
    """Thin vLLM translation layer for agent creator calls."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMAgentCreatorConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.last_request: dict[str, Any] | None = None
        self.last_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    def run_orchestrator(
        self,
        request: CreatorOrchestratorRequest,
        *,
        max_tool_calls: int,
    ) -> CreatorOrchestratorResponse:
        """Return one structured creator mutation plan."""

        response = self._client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": request.instructions},
                self._user_message(request),
            ],
            response_format=json_schema_response_format(
                name="agent_creator_plan",
                schema=agent_creator_orchestrator_output_schema(max_tool_calls),
            ),
        )
        self.last_request = self._client.last_request
        content = chat_message_optional_content(response)
        if content is None:
            response_id = chat_response_metadata(response).get("response_id")
            raise RuntimeError(
                "vLLM agent creator orchestrator response did not include "
                f"content for response {response_id!r}"
            )
        plan = parse_creator_orchestrator_plan_output(
            content,
            max_mutations=max_tool_calls,
        )
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **chat_response_metadata(response),
        }
        self.last_response_text = content
        self.last_response_metadata = metadata
        return CreatorOrchestratorResponse(
            text=content,
            tool_call_count=len(plan.mutations),
            mutations=plan.mutations,
            metadata={**request.metadata, **metadata},
        )

    def author_role(
        self,
        request: RoleAuthorRequest,
    ) -> AgentCreatorProviderResponse:
        """Call vLLM and return raw role-definition JSON text."""

        return self._structured_chat(request, text=request.text)

    def _user_message(
        self,
        request: CreatorOrchestratorRequest,
    ) -> dict[str, Any]:
        content = [
            {
                "type": "text",
                "text": request.text,
            }
        ]
        content.extend(
            vllm_image_content(
                [image.image for image in request.images],
                detail=self.config.input_image_detail,
                size=None,
                resample=self.config.input_image_resample,
                mime_type=self.config.image_mime_type,
            )
        )
        return {"role": "user", "content": content}

    def repair_role(
        self,
        request: RoleAuthorRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> AgentCreatorProviderResponse:
        """Ask vLLM to repair invalid role-definition JSON."""

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
        response = self._client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": text},
            ],
            response_format=json_schema_response_format(
                name="agent_creator_role",
                schema=request.output_schema,
            ),
        )
        self.last_request = self._client.last_request
        return self._provider_response(request, response)

    def _provider_response(
        self,
        request: RoleAuthorRequest,
        response: Any,
    ) -> AgentCreatorProviderResponse:
        content = chat_message_optional_content(response)
        metadata = {
            "backend": self.config.backend,
            "model": self.config.model,
            **chat_response_metadata(response),
        }
        self.last_response_text = content
        self.last_response_metadata = metadata
        if content is None:
            response_id = metadata.get("response_id")
            raise RuntimeError(
                "vLLM agent creator response did not include content "
                f"for response {response_id!r}"
            )
        return AgentCreatorProviderResponse(
            text=content,
            metadata={**request.metadata, **metadata},
        )
