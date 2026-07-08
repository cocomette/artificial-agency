"""OpenAI provider adapter for orchestrator agent X."""

from __future__ import annotations

import json
import os
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
from face_of_agi.models.orchestrator_agent.config import OpenAIOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    build_decision_prompt,
    load_agent_instructions,
    object_get,
    observation_images,
    openai_image_content,
    openai_tool_definitions,
    tool_result_feedback,
)


class OpenAIOrchestratorAgentAdapter(OrchestratorAgentAdapter):
    """Agent X adapter backed by OpenAI Responses."""

    def __init__(
        self,
        config: OpenAIOrchestratorAgentConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        resolved_config = config or OpenAIOrchestratorAgentConfig()
        self.provider = OpenAIOrchestratorAgentProvider(resolved_config, client=client)
        super().__init__(provider=self.provider, config=resolved_config)


class OpenAIOrchestratorAgentProvider:
    """Thin OpenAI translation layer for the shared Agent X loop."""

    backend = "openai"

    def __init__(
        self,
        config: OpenAIOrchestratorAgentConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = client
        self.instructions = load_agent_instructions()
        self.input_items: list[Any] = []
        self.tools: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial OpenAI Responses input for one X turn."""

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
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            openai_image_content(images, detail=self.config.input_image_detail)
        )
        self.input_items = [{"role": "user", "content": content}]
        self.tools = openai_tool_definitions(request.available_tools)

    def call(self) -> AgentProviderResponse:
        """Call OpenAI once and normalize function calls."""

        response = self._create_response()
        output = list(object_get(response, "output", []) or [])
        self.input_items.extend(output)
        function_calls = tuple(
            ProviderFunctionCall(
                name=str(object_get(item, "name", "")),
                arguments=object_get(item, "arguments", "{}"),
                call_id=str(object_get(item, "call_id", object_get(item, "id", ""))),
            )
            for item in output
            if object_get(item, "type") == "function_call"
        )
        usage = self._plain(object_get(response, "usage"))
        return AgentProviderResponse(
            function_calls=function_calls,
            response_id=(
                str(object_get(response, "id"))
                if object_get(response, "id") is not None
                else None
            ),
            usage=usage,
        )

    def append_tool_feedback(self, feedback: ProviderToolFeedback) -> None:
        """Append one orchestration-executed tool result."""

        self.input_items.append(
            {
                "type": "function_call_output",
                "call_id": feedback.call_id or "",
                "output": json.dumps(
                    tool_result_feedback(feedback.invocation),
                    sort_keys=True,
                ),
            }
        )
        try:
            image_content = openai_image_content(
                [feedback.invocation.tool_result.predicted_observation],
                detail=self.config.input_image_detail,
            )
        except Exception:
            return
        self.input_items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Tool result image for "
                            f"{feedback.invocation.observation_ref.memory}:"
                            f"{feedback.invocation.observation_ref.id}"
                        ),
                    },
                    *image_content,
                ],
            }
        )

    def append_repair(
        self,
        error: str,
        action_space: Sequence[ActionSpec],
    ) -> None:
        """Append one repair instruction to the OpenAI conversation."""

        allowed = ", ".join(action.name for action in action_space)
        self.input_items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Invalid response: "
                            f"{error}. Repair by using native tools only. "
                            f"Allowed final actions: {allowed}."
                        ),
                    }
                ],
            }
        )

    def _create_response(self) -> Any:
        request: dict[str, Any] = {
            "model": self.config.model,
            "instructions": self.instructions,
            "input": list(self.input_items),
            "tools": self.tools,
        }
        self._set_optional(request, "reasoning", self.config.reasoning)
        self._set_optional(request, "max_output_tokens", self.config.max_output_tokens)
        self._set_optional(request, "temperature", self.config.temperature)
        self._set_optional(request, "top_p", self.config.top_p)
        self._set_optional(request, "text", self.config.text)
        self._set_optional(request, "metadata", self.config.metadata)
        self._set_optional(request, "store", self.config.store)
        self._set_optional(request, "service_tier", self.config.service_tier)
        request.update(self.config.extra_request_options)
        self.last_request = request
        return self._require_client().responses.create(**request)

    def _require_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            self._set_optional(kwargs, "api_key", self._resolved_api_key())
            self._set_optional(kwargs, "base_url", self.config.base_url)
            self._set_optional(kwargs, "organization", self.config.organization)
            self._set_optional(kwargs, "project", self.config.project)
            self._set_optional(kwargs, "timeout", self.config.timeout)
            self._set_optional(kwargs, "max_retries", self.config.max_retries)
            self._set_optional(kwargs, "default_headers", self.config.default_headers)
            self._set_optional(kwargs, "default_query", self.config.default_query)
            self._client = OpenAI(**kwargs)
        return self._client

    def _resolved_api_key(self) -> str | None:
        if self.config.api_key:
            return self.config.api_key
        if self.config.api_key_env:
            return os.environ.get(self.config.api_key_env)
        return None

    def _set_optional(self, target: dict[str, Any], key: str, value: Any) -> None:
        if value is None or value == {} or value == []:
            return
        target[key] = value

    def _plain(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return {key: self._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._plain(item) for item in value]
        return value
