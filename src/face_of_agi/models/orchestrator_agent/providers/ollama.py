"""Ollama provider adapter for orchestrator agent X."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from face_of_agi.contracts import ActionSpec, RoleContext
from face_of_agi.debug.capture import capture_ollama_model_input
from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderStep,
    AgentToolSpec,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderFunctionCall,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import OllamaOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    build_agent_instructions,
    build_decision_prompt,
    final_action_repair_prompt,
    final_action_schema,
    function_call_name_and_arguments,
    object_get,
    observation_images,
    tool_result_feedback,
)
from face_of_agi.models.image_inputs import ollama_image_payloads
from face_of_agi.models.providers.ollama import (
    OllamaChatClient,
    object_get as ollama_object_get,
    response_usage,
    structured_json_content,
    structured_json_instructions,
)
from face_of_agi.runtime import timing as runtime_timing


class OllamaOrchestratorAgentAdapter(OrchestratorAgentAdapter):
    """Agent X adapter backed by a local Ollama chat model."""

    def __init__(
        self,
        config: OllamaOrchestratorAgentConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        resolved_config = config or OllamaOrchestratorAgentConfig()
        self.provider = OllamaOrchestratorAgentProvider(resolved_config, client=client)
        super().__init__(provider=self.provider, config=resolved_config)


class OllamaOrchestratorAgentProvider:
    """Thin Ollama translation layer for the shared Agent X loop."""

    backend = "ollama"

    def __init__(
        self,
        config: OllamaOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = OllamaChatClient(config, client=client)
        self.instructions = build_agent_instructions(RoleContext())
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial Ollama chat messages for one X turn."""

        self.instructions = build_agent_instructions(request.context)
        with runtime_timing.span("agent_x.build_prompt"):
            prompt = build_decision_prompt(
                world_game_context=request.world_game_context,
                goal_game_context=request.goal_game_context,
                first_observation=request.first_observation,
                current_observation=request.current_observation,
                action_space=request.action_space,
                recent_action_history=request.recent_action_history,
            )
        with runtime_timing.span("agent_x.observation_images"):
            images = observation_images(
                first_observation=request.first_observation,
                current_observation=request.current_observation,
                frame_scale=self.config.frame_scale,
            )
        self.messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": prompt,
                "images": self._image_payloads(images),
            },
        ]
        with runtime_timing.span("agent_x.tool_definitions"):
            self.tools = []

    def step(
        self,
        action_space: Sequence[ActionSpec],
        tool_specs: Sequence[AgentToolSpec],
    ) -> AgentProviderStep:
        """Call Ollama once and normalize tool calls/final output."""

        self.tools = self._tool_definitions(tool_specs)
        schema = final_action_schema(action_space)
        response = self._chat(schema)
        message = ollama_object_get(response, "message", {})
        provider_tool_calls = list(object_get(message, "tool_calls", []) or [])
        self.messages.append(self._assistant_message(message))
        tool_calls = []
        for provider_call in provider_tool_calls:
            name, arguments = function_call_name_and_arguments(provider_call)
            tool_calls.append(
                ProviderFunctionCall(name=name, arguments=arguments, call_id=None)
            )
        return AgentProviderStep(
            tool_calls=tuple(tool_calls),
            final_output=self._final_output(response),
            usage=response_usage(response),
        )

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Append one orchestration-executed tool result."""

        message: dict[str, Any] = {
            "role": "tool",
            "tool_name": feedback.invocation.tool_result.tool,
            "content": json.dumps(
                tool_result_feedback(feedback.invocation),
                sort_keys=True,
            ),
        }
        self.messages.append(message)

    def _messages_with_structured_instructions(
        self,
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages = list(self.messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": structured_json_instructions(
                    self.instructions,
                    schema,
                ),
            }
        return messages

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
        with runtime_timing.span("agent_x.ollama_chat"):
            response = self._client.chat(
                model=self.config.model,
                messages=self._messages_with_structured_instructions(schema),
                tools=self.tools or None,
                response_format=schema,
            )
        self.last_request = self._client.last_request
        self._capture_request(phase="final_action", response=response)
        return response

    def _tool_definitions(
        self,
        tool_specs: Sequence[AgentToolSpec],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in tool_specs
        ]

    def _final_output(self, response: Any) -> str | None:
        message = ollama_object_get(response, "message", {})
        content = object_get(message, "content", "") or ""
        if not str(content).strip():
            return None
        return structured_json_content(response)

    def _image_payloads(self, images: Sequence[Any]) -> list[str]:
        with runtime_timing.span("agent_x.image_encode", image_count=len(images)):
            return ollama_image_payloads(
                images,
                size=self.config.input_image_size,
                resample=self.config.input_image_resample,
            )

    def _assistant_message(self, message: Any) -> dict[str, Any]:
        content = object_get(message, "content", "") or ""
        tool_calls = object_get(message, "tool_calls", None)
        result: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            result["tool_calls"] = tool_calls
        thinking = object_get(message, "thinking", None)
        if thinking:
            result["thinking"] = thinking
        return result

    def _capture_request(self, *, phase: str, response: Any | None) -> None:
        if self.last_request is None:
            return
        capture_ollama_model_input(
            self,
            call_slot="agent",
            provider=self.backend,
            model=self.model,
            phase=phase,
            request=self.last_request,
            response=response,
        )
