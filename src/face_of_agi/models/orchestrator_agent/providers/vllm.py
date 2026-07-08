"""vLLM provider adapter for orchestrator agent X."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from face_of_agi.contracts import ActionSpec
from face_of_agi.debug.capture import capture_vllm_model_input
from face_of_agi.models.image_inputs import vllm_image_content
from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderStep,
    AgentToolSpec,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import VLLMOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    build_agent_instructions,
    build_decision_prompt,
    final_action_repair_prompt,
    final_action_schema,
    object_get,
    observation_images,
)
from face_of_agi.models.providers.openai import plain
from face_of_agi.models.providers.vllm import (
    VLLMChatClient,
    chat_message,
    chat_message_content,
    chat_response_metadata,
    json_schema_response_format,
)


class VLLMOrchestratorAgentAdapter(OrchestratorAgentAdapter):
    """Agent X adapter backed by vLLM Chat Completions."""

    def __init__(
        self,
        config: VLLMOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.model:
            raise ValueError("vLLM Agent X requires an explicit model")
        if config.max_tool_calls > 0:
            raise ValueError("vLLM Agent X does not support tool calls")
        self.provider = VLLMOrchestratorAgentProvider(config, client=client)
        super().__init__(provider=self.provider, config=config)


class VLLMOrchestratorAgentProvider:
    """Thin vLLM translation layer for the shared Agent X loop."""

    backend = "vllm"

    def __init__(
        self,
        config: VLLMOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = VLLMChatClient(config, client=client)
        self.instructions = ""
        self.messages: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial vLLM chat messages for one X turn."""

        self.instructions = build_agent_instructions(
            allowed_actions=request.action_space
        )
        prompt = build_decision_prompt(
            context=request.context,
            action_space=request.action_space,
            recent_action_history=request.recent_action_history,
            recent_action_history_available=(
                request.recent_action_history_available
            ),
        )
        images = observation_images(
            current_observation=request.current_observation,
        )
        self.messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *vllm_image_content(
                        images,
                        detail=self.config.input_image_detail,
                        size=self.config.input_image_size,
                        resample=self.config.input_image_resample,
                        mime_type=self.config.image_mime_type,
                    ),
                ],
            },
        ]

    def step(
        self,
        action_space: Sequence[ActionSpec],
        tool_specs: Sequence[AgentToolSpec],
    ) -> AgentProviderStep:
        """Call vLLM once and normalize final structured output."""

        del tool_specs
        schema = final_action_schema(action_space)
        response = self._chat(schema)
        message = chat_message(response)
        self.messages.append(self._assistant_message(message))
        return AgentProviderStep(
            final_output=chat_message_content(response),
            response_id=(
                str(object_get(response, "id"))
                if object_get(response, "id") is not None
                else None
            ),
            usage=plain(object_get(response, "usage")),
        )

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Reject tool feedback because vLLM Agent X exposes no tools."""

        del feedback
        raise RuntimeError("vLLM Agent X does not support tool calls")

    def append_repair(
        self,
        *,
        validation_error: str,
        action_space: Sequence[ActionSpec],
        invalid_text: str | None,
        attempt: int,
    ) -> None:
        """Append one structured-output repair instruction."""

        self.messages.append(
            {
                "role": "user",
                "content": final_action_repair_prompt(
                    action_space,
                    validation_error=validation_error,
                    invalid_text=invalid_text,
                    attempt=attempt,
                ),
            }
        )

    def _chat(self, schema: dict[str, Any]) -> Any:
        response = self._client.chat(
            model=self.config.model,
            messages=list(self.messages),
            response_format=json_schema_response_format(
                name="agent_final_action",
                schema=schema,
            ),
        )
        self.last_request = self._client.last_request
        self._capture_request(phase="final_action", response=response)
        return response

    def _assistant_message(self, message: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": object_get(message, "content", "") or "",
        }

    def _capture_request(self, *, phase: str, response: Any | None) -> None:
        if self.last_request is None:
            return
        capture_vllm_model_input(
            self,
            call_slot="agent",
            provider=self.backend,
            model=self.model,
            phase=phase,
            request=self.last_request,
            response=response,
            metadata={
                "response_metadata": chat_response_metadata(response),
            },
        )
