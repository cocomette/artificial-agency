"""OpenAI provider adapter for orchestrator agent X."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from face_of_agi.contracts import ActionSpec
from face_of_agi.debug.capture import capture_openai_model_input
from face_of_agi.models.orchestrator_agent.adapter import (
    AgentProviderStep,
    AgentToolSpec,
    AgentTurnRequest,
    OrchestratorAgentAdapter,
    ProviderFunctionCall,
    ProviderToolFeedback,
)
from face_of_agi.models.orchestrator_agent.config import OpenAIOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.tooling import (
    build_agent_instructions,
    build_decision_prompt,
    final_action_repair_prompt,
    final_action_schema,
    object_get,
    observation_images,
    openai_final_action_text_format,
    tool_result_feedback,
)
from face_of_agi.models.image_inputs import openai_image_content
from face_of_agi.models.providers.openai import (
    OpenAIResponsesClient,
    plain,
    response_output_text,
)
from face_of_agi.models.structured_output import append_output_schema_to_instructions


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
        self._responses = OpenAIResponsesClient(config, client=client)
        self.instructions = ""
        self.input_items: list[Any] = []
        self.tools: list[dict[str, Any]] = []
        self.last_request: dict[str, Any] | None = None

    def begin(self, request: AgentTurnRequest) -> None:
        """Build the initial OpenAI Responses input for one X turn."""

        self.instructions = append_output_schema_to_instructions(
            build_agent_instructions(glossary_actions=request.glossary_actions),
            final_action_schema(
                request.action_space,
                action6_targeting_mode=self.config.action6_targeting_mode,
            ),
            include=self.config.include_output_schema_in_instructions,
        )
        prompt = build_decision_prompt(
            context=request.context,
            action_space=request.action_space,
            recent_action_history=request.recent_action_history,
            recent_action_history_available=(
                request.recent_action_history_available
            ),
            action_outcome_evidence=request.action_outcome_evidence,
            game_memory=request.game_memory,
            crop_box_normalized=self.config.input_image_crop_box_normalized,
            action6_targeting_mode=self.config.action6_targeting_mode,
        )
        images = observation_images(
            current_observation=request.current_observation,
            frame_scale=self.config.frame_scale,
        )
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            openai_image_content(
                images,
                detail=self.config.input_image_detail,
                size=self.config.input_image_size,
                resample=self.config.input_image_resample,
                crop_box_normalized=self.config.input_image_crop_box_normalized,
            )
        )
        self.input_items = [{"role": "user", "content": content}]
        self.tools = []

    def step(
        self,
        action_space: Sequence[ActionSpec],
        tool_specs: Sequence[AgentToolSpec],
    ) -> AgentProviderStep:
        """Call OpenAI once and normalize tool calls/final output."""

        self.tools = self._tool_definitions(tool_specs)
        response = self._create_response(action_space)
        output = list(object_get(response, "output", []) or [])
        self.input_items.extend(output)
        final_output = response_output_text(response)
        tool_calls: list[ProviderFunctionCall] = []
        for item in output:
            if object_get(item, "type") != "function_call":
                continue
            name = str(object_get(item, "name", ""))
            arguments = object_get(item, "arguments", "{}")
            if name == "submit_action":
                final_output = arguments
                continue
            tool_calls.append(
                ProviderFunctionCall(
                    name=name,
                    arguments=arguments,
                    call_id=str(
                        object_get(item, "call_id", object_get(item, "id", ""))
                    ),
                )
            )
        usage = plain(object_get(response, "usage"))
        return AgentProviderStep(
            tool_calls=tuple(tool_calls),
            final_output=final_output,
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

    def append_repair(
        self,
        *,
        validation_error: str,
        action_space: Sequence[ActionSpec],
        invalid_text: str | None,
        attempt: int,
    ) -> None:
        """Append one structured-output repair instruction."""

        self._append_missing_function_call_outputs(validation_error)
        self.input_items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": final_action_repair_prompt(
                            action_space,
                            validation_error=validation_error,
                            invalid_text=invalid_text,
                            attempt=attempt,
                            action6_targeting_mode=(
                                self.config.action6_targeting_mode
                            ),
                        ),
                    }
                ],
            }
        )

    def _append_missing_function_call_outputs(self, error: str) -> None:
        """Satisfy Responses function calls that were rejected locally."""

        pending_call_ids: list[str] = []
        answered_call_ids: set[str] = set()
        for item in self.input_items:
            item_type = object_get(item, "type")
            call_id = object_get(item, "call_id", object_get(item, "id", None))
            if call_id is None:
                continue
            call_id = str(call_id)
            if item_type == "function_call":
                pending_call_ids.append(call_id)
            elif item_type == "function_call_output":
                answered_call_ids.add(call_id)

        output = json.dumps(
            {
                "ok": False,
                "error": error,
            },
            sort_keys=True,
        )
        for call_id in pending_call_ids:
            if call_id in answered_call_ids:
                continue
            self.input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            answered_call_ids.add(call_id)

    def _create_response(self, action_space: Sequence[ActionSpec]) -> Any:
        response = self._responses.create_response(
            model=self.config.model,
            instructions=self.instructions,
            input_items=list(self.input_items),
            tools=self.tools,
            text=openai_final_action_text_format(
                action_space,
                action6_targeting_mode=self.config.action6_targeting_mode,
            ),
            include_max_tool_calls=False,
        )
        self.last_request = self._responses.last_request
        self._capture_request(phase="final_action", response=response)
        return response

    def _tool_definitions(
        self,
        tool_specs: Sequence[AgentToolSpec],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in tool_specs
        ]

    def _capture_request(self, *, phase: str, response: Any | None) -> None:
        if self.last_request is None:
            return
        capture_openai_model_input(
            self,
            call_slot="agent",
            provider=self.backend,
            model=self.model,
            phase=phase,
            request=self.last_request,
            response=response,
        )
