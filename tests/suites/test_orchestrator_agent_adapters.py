"""Tests for OpenAI and Ollama orchestrator-agent backends."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from arcengine import GameAction
from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    ChangeSummaryElement,
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
    AgentToolSpec,
    OllamaOrchestratorAgentConfig,
    OpenAIOrchestratorAgentConfig,
)
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.change.components import arc_rendered_color_map
from face_of_agi.models.orchestrator_agent.tooling import (
    AgentOutputError,
    build_decision_prompt,
    final_action_schema,
    parse_action,
)


class FakeRuntime:
    """Small AgentToolRuntime test double."""

    def __init__(
        self,
        *,
        available_tools: tuple[str, ...] = ("world",),
        action_history: tuple[ActionHistoryEntry, ...] = (),
        previous_ref: ObservationRef | None = None,
    ) -> None:
        self.current_observation_ref = ObservationRef(memory="state", id="obs-current")
        self.previous_observation_ref = previous_ref
        self.first_observation_ref = ObservationRef(memory="state", id="obs-first")
        self.turn_id = 1
        self.current_source_state_id = 7
        self._available_tools = available_tools
        self._action_history = action_history
        self.calls: list[ToolCall] = []

    def available_observation_refs(self) -> tuple[ObservationRef, ...]:
        refs = [
            self.first_observation_ref,
            self.previous_observation_ref,
            self.current_observation_ref,
        ]
        return tuple(ref for ref in refs if ref is not None)

    def recent_action_history(self) -> tuple[ActionHistoryEntry, ...]:
        return self._action_history

    def available_tools(self) -> tuple[str, ...]:
        return self._available_tools

    def available_tool_specs(self) -> tuple[AgentToolSpec, ...]:
        return tuple(
            AgentToolSpec(
                name=name,
                description=f"{name} tool",
                parameters={"type": "object", "properties": {}},
            )
            for name in self._available_tools
        )

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
            output=Image.new("RGB", (64, 64), color=(10, 20, 30)),
            source_observation_ref=self.current_observation_ref,
            source_state_id=call.source_state_id,
            action=call.action,
        )
        return ExperimentToolInvocationResult(
            tool_result=result,
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


def _previous_observation() -> Observation:
    return Observation(
        id="obs-prev",
        step=1,
        frame=Image.new("RGB", (8, 8), color=(127, 127, 127)),
    )


def _input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        content
        for content in request["input"][0]["content"]
        if content.get("type") == "input_image"
    ]


def _all_input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for item in request["input"]:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if content.get("type") == "input_image":
                images.append(content)
    return images


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _tool_names(request: dict[str, Any]) -> list[str]:
    return [tool["name"] for tool in request.get("tools", [])]


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


def test_openai_agent_captures_provider_reasoning_summaries() -> None:
    first, current = _observations()
    runtime = FakeRuntime()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        summary=[
                            SimpleNamespace(
                                type="summary_text",
                                text="considered a world-tool probe",
                            )
                        ],
                    ),
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
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "used the prediction to choose ACTION1",
                            }
                        ],
                    },
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
        OpenAIOrchestratorAgentConfig(
            max_tool_calls=2,
            repair_attempts=1,
            reasoning={"effort": "low", "summary": "auto"},
        ),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(game="use tools briefly"),
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.reasoning_summary is None
    assert runtime.calls[0].tool == "world"
    assert decision.trace.tool_calls == runtime.calls
    assert decision.trace.metadata["backend"] == "openai"
    assert decision.trace.metadata["repair_count"] == 0
    assert decision.trace.metadata["tool_call_count"] == 1
    assert decision.trace.metadata["provider_response_ids"] == ["resp-1", "resp-2"]
    assert len(adapter.last_provider_requests) == 2
    assert adapter.last_provider_requests[0]["model"] == "gpt-5-nano"
    assert adapter.last_provider_requests[0]["input"][0]["role"] == "user"
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["reasoning"] == {
        "effort": "low",
        "summary": "auto",
    }
    first_prompt = client.responses.calls[0]["input"][0]["content"][0]["text"]
    assert "## Agent context" in first_prompt
    assert "use tools briefly" in first_prompt
    assert "## Recent actions\n\nnone" in first_prompt
    assert len(_input_images(client.responses.calls[0])) == 1
    second_input = client.responses.calls[1]["input"]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in second_input
    )
    assert any(
        (
            (isinstance(item, dict) and item.get("type") == "reasoning")
            or getattr(item, "type", None) == "reasoning"
        )
        for item in second_input
    )
    assert any(
        content.get("type") == "input_image"
        for item in second_input
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for content in item["content"]
    )


def test_openai_agent_falls_back_to_submit_action_reasoning_summary() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.trace.reasoning_summary is None
    assert decision.trace.metadata["tool_call_count"] == 0
    assert decision.trace.metadata["repair_count"] == 0


def test_openai_agent_hides_model_tools_when_tool_budget_is_zero() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    adapter.decide(
        RoleContext(general="agent K", game="agent L"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=("world", "goal")),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    prompt = request["input"][0]["content"][0]["text"]
    assert _tool_names(request) == []
    assert "## Allowed actions" in prompt


def test_openai_agent_prompt_includes_recent_action_history() -> None:
    first, current = _observations()
    history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=5,
        changed_pixel_percent=7.8125,
        completed_levels=1,
        action_count=3,
        change_summary="white area expanded",
    )
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(general="agent K", game="agent L"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(history_entry,),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = client.responses.calls[0]["input"][0]["content"][0]["text"]
    system_instructions = client.responses.calls[0]["instructions"]
    normalized_system_instructions = " ".join(system_instructions.split())
    assert "observed facts from the frame and recent transitions win" in (
        normalized_system_instructions
    )
    assert "Prior `ACTION6` rows in recent actions are rendered as target" in (
        system_instructions
    )
    assert "agent K" not in system_instructions
    assert "agent L" not in system_instructions
    assert "agent K\n\nagent L" in prompt
    assert "instructions" not in prompt
    assert "- ACTION1" in prompt
    assert "[changed_pixels=5]" in prompt
    assert "[changed_area=7.8125%]" in prompt
    assert "[completed_levels=1]" in prompt
    assert "[action_count=3]" in prompt
    assert "change: white area expanded" in prompt


def test_openai_agent_prompt_includes_change_element_history() -> None:
    first, current = _observations()
    history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=5,
        changed_pixel_percent=7.8125,
        completed_levels=1,
        action_count=3,
        change_summary="legacy fallback",
        change_elements=(
            ChangeSummaryElement(
                element_name="door",
                element_description="green door on the right edge",
                element_mutation="opened after ACTION1",
            ),
        ),
    )
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(general="agent K", game="agent L"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(history_entry,),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    prompt = request["input"][0]["content"][0]["text"]
    system_instructions = request["instructions"]
    assert "Elements and associated changes:" in prompt
    assert "- door: green door on the right edge; mutations: opened after ACTION1" in (
        prompt
    )
    assert "legacy fallback" not in prompt
    assert "Elements may be targets, triggers, objects, characters" in (
        system_instructions
    )


def test_agent_prompt_omits_recent_action_history_reasoning() -> None:
    first, current = _observations()
    history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=0,
        change_summary="no changes",
    )
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(history_entry,),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = client.responses.calls[0]["input"][0]["content"][0]["text"]
    assert "ACTION1 [latest] [changed_pixels=0]" in prompt
    assert "change: First and final frames are identical." in prompt


def test_openai_agent_prompt_includes_current_observation_image() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    prompt = request["input"][0]["content"][0]["text"]
    assert "## Agent context" in prompt
    assert len(_input_images(request)) == 1


def test_openai_agent_applies_configured_input_image_size_to_images() -> None:
    first, current = _observations()
    runtime = FakeRuntime()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-world",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="world",
                        call_id="call-world",
                        arguments=_world_arguments(),
                    )
                ],
            ),
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(
            input_image_size="10x12",
            input_image_resample="nearest",
            max_tool_calls=1,
            repair_attempts=0,
        ),
        client=client,
    )

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    initial_images = _input_images(client.responses.calls[0])
    initial_sizes = [
        _decode_data_url_image(image["image_url"]).size
        for image in initial_images
    ]
    assert initial_sizes == [
        (10, 12),
    ]

    all_feedback_call_images = _all_input_images(client.responses.calls[1])
    assert _decode_data_url_image(all_feedback_call_images[-1]["image_url"]).size == (
        10,
        12,
    )


def test_openai_agent_uses_single_current_image_when_previous_matches() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            )
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    prompt = request["input"][0]["content"][0]["text"]
    assert "## Recent actions\n\nnone" in prompt
    assert len(_input_images(request)) == 1


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
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["repair_count"] == 1
    repair_input = client.responses.calls[1]["input"]
    missing_output = [
        item
        for item in repair_input
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call-bad"
    ][0]
    assert json.loads(missing_output["output"])["ok"] is False
    repair_text = client.responses.calls[1]["input"][-1]["content"][0]["text"]
    assert "Repair attempt 1" in repair_text
    assert "Validation error:" in repair_text


def test_openai_agent_repairs_over_budget_tool_call_with_function_output() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            SimpleNamespace(
                id="resp-tool",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="world",
                        call_id="call-world",
                        arguments=_world_arguments(),
                    )
                ],
            ),
            SimpleNamespace(
                id="resp-submit",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="submit_action",
                        call_id="call-submit",
                        arguments=_submit_arguments(),
                    )
                ],
            ),
        ]
    )
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=1),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    repair_input = client.responses.calls[1]["input"]
    rejected_outputs = [
        item
        for item in repair_input
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call-world"
    ]
    assert decision.final_action.action_id == "ACTION1"
    assert rejected_outputs
    output_error = json.loads(rejected_outputs[0]["output"])["error"]
    assert "tool-call budget" in output_error


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

    with pytest.raises(RuntimeError, match="tool-call budget"):
        adapter.decide(
            RoleContext(),
            current,
            [ActionSpec(action_id="ACTION1")],
            FakeRuntime(),
            first_observation_ref=ObservationRef(memory="state", id=first.id),
        )


def test_final_action_schema_requires_complex_action_target() -> None:
    schema = final_action_schema((ActionSpec(GameAction.ACTION6),))
    action_schema = schema["properties"]["action"]
    data_schema = action_schema["properties"]["data"]

    assert data_schema["type"] == "object"
    assert data_schema["required"] == ["x", "y"]
    assert action_schema["properties"]["target"]["type"] == "string"
    assert action_schema["required"] == ["action_id", "data", "target"]


def test_final_action_schema_bbox_color_mode_requires_target_bbox_and_color() -> None:
    schema = final_action_schema(
        (ActionSpec(GameAction.ACTION6),),
        action6_targeting_mode="bbox_color",
    )
    action_schema = schema["properties"]["action"]

    assert "data" not in action_schema["properties"]
    assert action_schema["properties"]["bbox"]["minItems"] == 4
    assert action_schema["properties"]["target_rgb_color"]["minItems"] == 3
    assert action_schema["required"] == [
        "action_id",
        "target",
        "bbox",
        "target_rgb_color",
    ]


def test_final_action_schema_bbox_center_mode_requires_target_and_bbox() -> None:
    schema = final_action_schema(
        (ActionSpec(GameAction.ACTION6),),
        action6_targeting_mode="bbox_center",
    )
    action_schema = schema["properties"]["action"]

    assert "data" not in action_schema["properties"]
    assert "target_rgb_color" not in action_schema["properties"]
    assert action_schema["properties"]["bbox"]["minItems"] == 4
    assert action_schema["required"] == [
        "action_id",
        "target",
        "bbox",
    ]


def test_parse_action_requires_action6_target() -> None:
    with pytest.raises(AgentOutputError, match="ACTION6 requires non-empty"):
        parse_action(
            {"action_id": "ACTION6", "data": {"x": 500, "y": 250}},
            (ActionSpec(GameAction.ACTION6),),
        )


def test_parse_action_stores_action6_target_and_rejects_simple_target() -> None:
    action = parse_action(
        {
            "action_id": "ACTION6",
            "data": {"x": 500, "y": 250},
            "target": " upper middle tile ",
        },
        (ActionSpec(GameAction.ACTION6),),
    )

    assert action.data == {"x": 32, "y": 16}
    assert action.target == "upper middle tile"

    with pytest.raises(AgentOutputError, match="simple actions must not include"):
        parse_action(
            {"action_id": "ACTION1", "target": "anything"},
            (ActionSpec(GameAction.ACTION1),),
        )


def test_parse_action_bbox_color_mode_retargets_action6_to_arc_grid() -> None:
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    red = arc_rendered_color_map()[8]
    image.putpixel((20, 30), red)
    observation = Observation(id="obs-current", step=1, frame=image)

    action = parse_action(
        {
            "action_id": "ACTION6",
            "target": "red pixel",
            "bbox": [0, 0, 500, 600],
            "target_rgb_color": list(red),
        },
        (ActionSpec(GameAction.ACTION6),),
        current_observation=observation,
        action6_targeting_mode="bbox_color",
    )

    assert action.data == {"x": 20, "y": 30}
    assert action.target == "red pixel"
    assert action.target_value == 8
    assert action.target_bbox == (0, 0, 500, 600)


def test_parse_action_bbox_center_mode_clicks_bbox_center() -> None:
    action = parse_action(
        {
            "action_id": "ACTION6",
            "target": "center region",
            "bbox": [250, 250, 750, 750],
        },
        (ActionSpec(GameAction.ACTION6),),
        action6_targeting_mode="bbox_center",
    )

    assert action.data == {"x": 32, "y": 32}
    assert action.target == "center region"
    assert action.target_value is None
    assert action.target_bbox == (250, 250, 750, 750)


def test_parse_action_bbox_center_mode_rejects_old_coordinate_data() -> None:
    with pytest.raises(AgentOutputError, match="unexpected keys: data"):
        parse_action(
            {
                "action_id": "ACTION6",
                "data": {"x": 500, "y": 250},
                "target": "old coordinate",
            },
            (ActionSpec(GameAction.ACTION6),),
            current_observation=_observations()[1],
            action6_targeting_mode="bbox_center",
        )


def test_parse_action_bbox_center_mode_rejects_rgb_retargeting_data() -> None:
    with pytest.raises(AgentOutputError, match="unexpected keys: target_rgb_color"):
        parse_action(
            {
                "action_id": "ACTION6",
                "target": "old color target",
                "bbox": [0, 0, 500, 500],
                "target_rgb_color": [255, 0, 0],
            },
            (ActionSpec(GameAction.ACTION6),),
            current_observation=_observations()[1],
            action6_targeting_mode="bbox_center",
        )


def test_agent_prompt_lists_bbox_center_action_shape_when_enabled() -> None:
    prompt = build_decision_prompt(
        context=RoleContext(),
        action_space=[ActionSpec(action_id=GameAction.ACTION6)],
        action6_targeting_mode="bbox_center",
    )

    assert "ACTION6(target,bbox)" in prompt


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
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert runtime.calls[0].tool == "world"
    assert decision.trace.metadata["backend"] == "ollama"
    assert decision.trace.metadata["repair_count"] == 0
    assert len(adapter.last_provider_requests) == 2
    assert adapter.last_provider_requests[0]["model"] == "gemma4:e4b"
    assert adapter.last_provider_requests[1]["messages"][-1]["role"] == "tool"
    assert client.calls[0]["model"] == "gemma4:e4b"
    assert client.calls[0].get("think") is not True
    assert any(message.get("role") == "tool" for message in client.calls[1]["messages"])
    tool_message = [
        message for message in client.calls[1]["messages"] if message.get("role") == "tool"
    ][0]
    assert json.loads(tool_message["content"])["tool"] == "world"


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
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.metadata["repair_count"] == 1
    assert any(
        message.get("role") == "user"
        and str(message.get("content", "")).startswith("Repair attempt 1")
        for message in client.calls[1]["messages"]
    )


def test_ollama_agent_falls_back_to_plain_text_submit_action() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {"action": {"action_id": "ACTION2", "data": None}}
                    ),
                    "tool_calls": [],
                },
                done_reason="stop",
            ),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION2"
    assert decision.trace.reasoning_summary is None
    assert decision.trace.metadata["repair_count"] == 0


def test_ollama_agent_falls_back_to_cheat_action_alias() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {"action": {"action_id": "ACTION4", "data": None}}
                    ),
                    "tool_calls": [],
                },
                done_reason="stop",
            ),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(
            game=(
                "Cheat action context from the local game source:\n"
                "ACTION1: up arrow\n"
                "ACTION2: down arrow\n"
                "ACTION3: left arrow\n"
                "ACTION4: right arrow"
            )
        ),
        current,
        [
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
            ActionSpec(action_id="ACTION3"),
            ActionSpec(action_id="ACTION4"),
        ],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION4"
    assert decision.trace.reasoning_summary is None


def test_ollama_agent_falls_back_to_json_content_submit_action() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": {"action_id": "ACTION2", "data": None},
                            "reasoning_summary": "try down",
                        }
                    ),
                    "tool_calls": [],
                },
                done_reason="stop",
            ),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(
            max_tool_calls=0,
            repair_attempts=0,
            format={"type": "object"},
        ),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION2"
    assert decision.trace.reasoning_summary is None
    assert client.calls[0]["format"]["required"] == ["action"]


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
        control_mode=FrameControlMode.animation_unroll((ActionSpec.none(),)),
    )

    assert frame_context.control_mode.allowed_actions[0].is_none()
