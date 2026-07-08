"""Tests for OpenAI and Ollama orchestrator-agent backends."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionSpec,
    ExperimentToolInvocationResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.models.orchestrator_agent import (
    OllamaOrchestratorAgentConfig,
    OpenAIOrchestratorAgentConfig,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
    openai_tool_definitions,
)


class FakeRuntime:
    """Small AgentToolRuntime test double."""

    def __init__(self, *, available_tools: tuple[str, ...] = ("world",)) -> None:
        self.current_observation_ref = ObservationRef(memory="state", id="obs-current")
        self.first_observation_ref = ObservationRef(memory="state", id="obs-first")
        self.turn_id = 1
        self._available_tools = available_tools
        self.calls: list[ToolCall] = []

    def available_observation_refs(self) -> tuple[ObservationRef, ...]:
        return (self.first_observation_ref, self.current_observation_ref)

    def available_tools(self) -> tuple[str, ...]:
        return self._available_tools

    def tool_metadata(self) -> dict[str, Any]:
        return {"tools_enabled": True, "available_tools": list(self._available_tools)}

    def invoke(
        self,
        call: ToolCall,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentToolInvocationResult:
        del metadata
        self.calls.append(call)
        result = ToolResult(
            id="world-result",
            tool=call.tool,
            predicted_observation=Image.new("RGB", (64, 64), color=(10, 20, 30)),
            source_observation_ref=call.observation_ref,
            action=call.action,
        )
        return ExperimentToolInvocationResult(
            tool_result=result,
            observation_ref=ObservationRef(memory="experimental", id="7"),
            experiment_record=SimpleNamespace(id=7),
        )


class FakeResponses:
    """Captures OpenAI Responses calls."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAIClient:
    """Tiny OpenAI client stand-in."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = FakeResponses(responses)


class FakeOllamaClient:
    """Tiny Ollama client stand-in."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _observations() -> tuple[Observation, Observation]:
    return (
        Observation(
            id="obs-first",
            step=0,
            frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ),
        Observation(
            id="obs-current",
            step=2,
            frame=Image.new("RGB", (8, 8), color=(255, 255, 255)),
        ),
    )


def _world_arguments() -> str:
    return json.dumps(
        {
            "observation_ref": {"memory": "state", "id": "obs-current"},
            "action": {"action_id": "ACTION1", "data": None},
        }
    )


def _submit_arguments(action_id: str = "ACTION1") -> str:
    return json.dumps(
        {
            "action": {"action_id": action_id, "data": None},
            "reasoning_summary": "selected a valid action",
        }
    )


def test_openai_agent_executes_tool_loop_and_submits_action() -> None:
    first, current = _observations()
    runtime = FakeRuntime()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="world",
                        call_id="call-world",
                        arguments=_world_arguments(),
                    )
                ],
                usage={"input_tokens": 10},
            ),
            SimpleNamespace(
                id="resp-2",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
                usage={"input_tokens": 20},
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=2, repair_attempts=1),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(game="use tools briefly"),
        first,
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
    )

    assert decision.final_action.action_id == "ACTION1"
    assert runtime.calls[0].tool == "world"
    assert decision.trace.tool_calls == runtime.calls
    assert decision.trace.metadata["backend"] == "openai"
    assert decision.trace.metadata["repair_count"] == 0
    assert decision.trace.metadata["provider_response_ids"] == ["resp-1", "resp-2"]
    assert len(client.responses.calls) == 2
    second_input = client.responses.calls[1]["input"]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second_input
    )
    assert any(
        content.get("type") == "input_image"
        for item in second_input
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for content in item["content"]
    )


def test_openai_agent_repairs_invalid_final_action_once() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-invalid",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-bad",
                        arguments=_submit_arguments("BAD"),
                    )
                ],
            ),
            SimpleNamespace(
                id="resp-valid",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-good",
                        arguments=_submit_arguments("ACTION1"),
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    decision = adapter.decide(
        RoleContext(),
        first,
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["repair_count"] == 1
    assert "Invalid response:" in client.responses.calls[1]["input"][-1]["content"][0]["text"]


def test_openai_agent_fails_when_tool_budget_is_exhausted() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="world",
                        call_id="call-world",
                        arguments=_world_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    with pytest.raises(RuntimeError, match="tool-call budget exhausted"):
        adapter.decide(
            RoleContext(),
            first,
            current,
            [ActionSpec(action_id="ACTION1")],
            FakeRuntime(),
        )


def test_openai_tool_schema_requires_complex_action_coordinates() -> None:
    tools = openai_tool_definitions(())
    submit_action = tools[0]

    data_schema = submit_action["parameters"]["properties"]["action"]["properties"][
        "data"
    ]

    assert data_schema["type"] == ["object", "null"]
    assert data_schema["required"] == ["x", "y"]


def test_ollama_agent_executes_tool_loop_and_submits_action() -> None:
    first, current = _observations()
    runtime = FakeRuntime()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "world",
                                "arguments": json.loads(_world_arguments()),
                            }
                        }
                    ],
                },
                done_reason="tool_calls",
                prompt_eval_count=10,
            ),
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_action",
                                "arguments": json.loads(_submit_arguments()),
                            }
                        }
                    ],
                },
                done_reason="stop",
                eval_count=5,
            ),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(max_tool_calls=2, repair_attempts=1),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(game="use local tools"),
        first,
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
    )

    assert decision.final_action.action_id == "ACTION1"
    assert runtime.calls[0].tool == "world"
    assert decision.trace.metadata["backend"] == "ollama"
    assert decision.trace.metadata["repair_count"] == 0
    assert client.calls[0]["model"] == "gemma4:e4b"
    assert client.calls[0]["think"] is False
    assert any(message.get("role") == "tool" for message in client.calls[1]["messages"])
    tool_message = [
        message for message in client.calls[1]["messages"] if message.get("role") == "tool"
    ][0]
    assert tool_message["images"]


def test_ollama_agent_repairs_invalid_final_action_once() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_action",
                                "arguments": json.loads(_submit_arguments("BAD")),
                            }
                        }
                    ]
                }
            ),
            SimpleNamespace(
                message={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_action",
                                "arguments": json.loads(_submit_arguments()),
                            }
                        }
                    ]
                }
            ),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(client=client)

    decision = adapter.decide(
        RoleContext(),
        first,
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["repair_count"] == 1
    assert client.calls[1]["messages"][-1]["content"].startswith("Invalid response:")


def test_fake_runtime_metadata_can_disable_animation_frame_tools() -> None:
    source = Observation(id="obs-current", step=0, frame={"frame": 0})
    ref = ObservationRef(memory="state", id=source.id)
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=source,
        frame_index=0,
        frame_count=2,
        control_mode=FrameControlMode.animation_unroll(),
    )

    assert frame_context.control_mode.allowed_actions[0].is_none()
