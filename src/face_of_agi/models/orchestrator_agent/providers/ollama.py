"""Ollama provider adapter for orchestrator agent X."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from face_of_agi.contracts import ActionSpec
from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderResponse,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderFunctionCall,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import OllamaOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    build_decision_prompt,
    function_call_name_and_arguments,
    load_agent_instructions,
    object_get,
    observation_images,
    ollama_image_payloads,
    ollama_tool_definitions,
    tool_result_feedback,
)


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
        self._client = client
        self.instructions = load_agent_instructions()
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial Ollama chat messages for one X turn."""

        prompt = build_decision_prompt(
            context=request.context,
            first_observation=request.first_observation,
            current_observation=request.current_observation,
            action_space=request.action_space,
            tool_runtime=request.tool_runtime,
        )
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
                "images": ollama_image_payloads(images),
            },
        ]
        self.tools = ollama_tool_definitions(request.available_tools)

    def call(self) -> AgentProviderResponse:
        """Call Ollama once and normalize function calls."""

        response = self._chat()
        message = object_get(response, "message", {})
        provider_tool_calls = list(object_get(message, "tool_calls", []) or [])
        self.messages.append(self._assistant_message(message))
        function_calls = []
        for provider_call in provider_tool_calls:
            name, arguments = function_call_name_and_arguments(provider_call)
            function_calls.append(
                ProviderFunctionCall(name=name, arguments=arguments, call_id=None)
            )
        return AgentProviderResponse(
            function_calls=tuple(function_calls),
            usage=self._response_usage(response),
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
        try:
            message["images"] = ollama_image_payloads(
                [feedback.invocation.tool_result.predicted_observation]
            )
        except Exception:
            pass
        self.messages.append(message)

    def append_repair(
        self,
        error: str,
        action_space: Sequence[ActionSpec],
    ) -> None:
        """Append one repair instruction to the Ollama conversation."""

        allowed = ", ".join(action.name for action in action_space)
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Invalid response: "
                    f"{error}. Repair by using native tools only. "
                    f"Allowed final actions: {allowed}."
                ),
            }
        )

    def _chat(self) -> Any:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(self.messages),
            "tools": self.tools,
            "stream": False,
            "think": self.config.think,
        }
        if self.config.options:
            request["options"] = self.config.options
        if self.config.keep_alive is not None:
            request["keep_alive"] = self.config.keep_alive
        self.last_request = request
        return self._require_client().chat(**request)

    def _require_client(self) -> Any:
        if self._client is None:
            import ollama

            if self.config.host:
                self._client = ollama.Client(host=self.config.host)
            else:
                self._client = ollama
        return self._client

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

    def _response_usage(self, response: Any) -> dict[str, Any]:
        keys = (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "done_reason",
        )
        return {
            key: object_get(response, key)
            for key in keys
            if object_get(response, key) is not None
        }
