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
    ActionHistoryResetMarker,
    ActionSpec,
    ExperimentToolInvocationResult,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.models.orchestrator_agent import (
    AgentProviderStep,
    AgentToolSpec,
    OrchestratorAgentAdapter,
    OrchestratorAgentConfig,
    OllamaOrchestratorAgentConfig,
    OpenAIOrchestratorAgentConfig,
    ProviderFunctionCall,
)
from face_of_agi.debug.capture import drain_model_input_debug_records
from face_of_agi.models.orchestrator_agent.providers.ollama import (
    OllamaOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.providers.openai import (
    OpenAIOrchestratorAgentAdapter,
)
from face_of_agi.models.orchestrator_agent.tooling import final_action_schema
from face_of_agi.models.orchestrator_agent.tooling import parse_action


class FakeRuntime:
    """Small AgentToolRuntime test double."""

    def __init__(
        self,
        *,
        available_tools: tuple[str, ...] = ("world",),
    ) -> None:
        self.current_observation_ref = ObservationRef(memory="state", id="obs-current")
        self.first_observation_ref = ObservationRef(memory="state", id="obs-first")
        self.current_source_state_id = 3
        self.turn_id = 1
        self._available_tools = available_tools
        self.calls: list[ToolCall] = []

    def available_tools(self) -> tuple[str, ...]:
        return self._available_tools

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


class FakeGenericRuntime(FakeRuntime):
    """AgentToolRuntime test double with generic Agent X tool specs."""

    def __init__(self) -> None:
        super().__init__(available_tools=())

    def available_tool_specs(self) -> tuple[AgentToolSpec, ...]:
        return (
            AgentToolSpec(
                name="inspect",
                description="Inspect a temporary state artifact.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_state_id": {"type": "integer"},
                    },
                    "required": ["source_state_id"],
                    "additionalProperties": False,
                },
            ),
        )


class ScriptedProvider:
    """Provider session test double for the shared Agent X loop."""

    backend = "openai"
    model = "gpt-5-nano"

    def __init__(self, steps: list[AgentProviderStep]) -> None:
        self.steps = steps
        self.last_request: dict[str, Any] | None = None
        self.tool_feedback: list[Any] = []
        self.repairs: list[str] = []
        self.seen_tool_specs: list[tuple[AgentToolSpec, ...]] = []

    def begin(self, request: Any) -> None:
        self.last_request = {"begin": True, "action_count": len(request.action_space)}

    def step(
        self,
        action_space: Any,
        tool_specs: tuple[AgentToolSpec, ...],
    ) -> AgentProviderStep:
        self.seen_tool_specs.append(tuple(tool_specs))
        self.last_request = {
            "tool_specs": [spec.name for spec in tool_specs],
            "schema": final_action_schema(action_space),
        }
        return self.steps.pop(0)

    def append_tool_feedback(self, feedback: Any) -> None:
        self.tool_feedback.append(feedback)

    def append_repair(
        self,
        *,
        validation_error: str,
        action_space: Any,
        invalid_text: str | None,
        attempt: int,
    ) -> None:
        del action_space
        self.repairs.append(
            json.dumps(
                {
                    "attempt": attempt,
                    "invalid_text": invalid_text,
                    "validation_error": validation_error,
                },
                sort_keys=True,
            )
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


def _input_images(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        content
        for content in request["input"][0]["content"]
        if content.get("type") == "input_image"
    ]


def _input_text(request: dict[str, Any]) -> str:
    return request["input"][0]["content"][0]["text"]


def _instructions(request: dict[str, Any]) -> str:
    return request["instructions"]


def _decode_data_url_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _submit_arguments(
    action_id: str = "ACTION1",
    *,
    data: dict[str, Any] | None = None,
    target: str | None = None,
) -> str:
    action: dict[str, Any] = {"action_id": action_id}
    if data is not None:
        action["data"] = data
    if target is not None:
        action["target"] = target
    return json.dumps(
        {
            "action": action,
        }
    )


def _openai_final_response(
    response_id: str = "resp-final",
    *,
    action_id: str = "ACTION1",
    usage: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        output=[],
        output_text=_submit_arguments(action_id),
        usage=usage,
    )


def _ollama_final_response(
    *,
    action_id: str = "ACTION1",
    done_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        message={
            "role": "assistant",
            "content": _submit_arguments(action_id),
            "tool_calls": [],
        },
        done_reason=done_reason,
    )


def test_shared_agent_loop_executes_tool_then_accepts_final_output() -> None:
    first, current = _observations()
    runtime = FakeGenericRuntime()
    provider = ScriptedProvider(
        [
            AgentProviderStep(
                tool_calls=(
                    ProviderFunctionCall(
                        name="inspect",
                        arguments={"source_state_id": 3},
                        call_id="call-1",
                    ),
                ),
                final_output=_submit_arguments("ACTION2"),
            ),
            AgentProviderStep(final_output=_submit_arguments("ACTION1")),
        ]
    )
    adapter = OrchestratorAgentAdapter(
        provider=provider,
        config=OrchestratorAgentConfig(max_tool_calls=1, repair_attempts=0),
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")],
        runtime,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert [call.tool for call in runtime.calls] == ["inspect"]
    assert [call.tool for call in decision.trace.tool_calls] == ["inspect"]
    assert [result.tool for result in decision.trace.tool_results] == ["inspect"]
    assert provider.tool_feedback[0].call_id == "call-1"
    assert provider.seen_tool_specs[0][0].name == "inspect"


def test_shared_agent_loop_rejects_tool_calls_over_budget() -> None:
    first, current = _observations()
    provider = ScriptedProvider(
        [
            AgentProviderStep(
                tool_calls=(
                    ProviderFunctionCall(
                        name="inspect",
                        arguments={"source_state_id": 3},
                    ),
                    ProviderFunctionCall(
                        name="inspect",
                        arguments={"source_state_id": 3},
                    ),
                ),
            )
        ]
    )
    adapter = OrchestratorAgentAdapter(
        provider=provider,
        config=OrchestratorAgentConfig(max_tool_calls=1, repair_attempts=0),
    )

    with pytest.raises(RuntimeError, match="tool-call budget"):
        adapter.decide(
            RoleContext(),
            current,
            [ActionSpec(action_id="ACTION1")],
            FakeGenericRuntime(),
            first_observation_ref=ObservationRef(memory="state", id=first.id),
        )


def test_shared_agent_loop_rejects_tools_when_budget_is_zero() -> None:
    first, current = _observations()
    provider = ScriptedProvider(
        [
            AgentProviderStep(
                tool_calls=(
                    ProviderFunctionCall(
                        name="inspect",
                        arguments={"source_state_id": 3},
                    ),
                ),
                final_output=_submit_arguments("ACTION1"),
            )
        ]
    )
    adapter = OrchestratorAgentAdapter(
        provider=provider,
        config=OrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
    )

    with pytest.raises(RuntimeError, match="no tools are available"):
        adapter.decide(
            RoleContext(),
            current,
            [ActionSpec(action_id="ACTION1")],
            FakeGenericRuntime(),
            first_observation_ref=ObservationRef(memory="state", id=first.id),
        )


def test_openai_agent_ignores_provider_reasoning_summaries() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            _openai_final_response(
                "resp-final",
                usage={"input_tokens": 30},
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
        RoleContext(game="choose directly"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=("world",)),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert decision.trace.reasoning_summary is None
    assert decision.trace.tool_calls == []
    assert decision.trace.metadata["backend"] == "openai"
    assert decision.trace.metadata["repair_count"] == 0
    assert decision.trace.metadata["provider_response_ids"] == ["resp-final"]
    assert len(adapter.last_provider_requests) == 1
    assert adapter.last_provider_requests[0]["model"] == "gpt-5-nano"
    assert adapter.last_provider_requests[0]["input"][0]["role"] == "user"
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["reasoning"] == {
        "effort": "low",
        "summary": "auto",
    }
    debug_records = drain_model_input_debug_records(adapter)
    assert debug_records[0]["call_slot"] == "agent"
    assert debug_records[0]["provider"] == "openai"
    assert debug_records[0]["phase"] == "final_action"
    assert debug_records[0]["request"]["input"][0]["role"] == "user"
    assert debug_records[0]["usage"] == {"input_tokens": 30}
    assert debug_records[0]["metadata"]["response_output_text"] == _submit_arguments()
    assert debug_records[0]["metadata"]["response_metadata"]["response_id"] == (
        "resp-final"
    )
    assert debug_records[0]["metadata"]["response_payload"]["output_text"] == (
        _submit_arguments()
    )
    first_prompt = _input_text(client.responses.calls[0])
    assert first_prompt.startswith("## Game context")
    assert "## Recent actions" in first_prompt
    assert "Attached frame" in _instructions(client.responses.calls[0])
    assert len(_input_images(client.responses.calls[0])) == 1
    assert client.responses.calls[0]["text"]["format"]["schema"]["properties"]["action"][
        "properties"
    ]["action_id"]["enum"] == ["ACTION1"]


def test_openai_agent_prompt_uses_only_allowed_actions_for_glossary() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(max_tool_calls=0, repair_attempts=0),
        client=client,
    )

    adapter.decide(
        RoleContext(general="agent K", game="agent L"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=("world", "goal")),
        glossary_actions=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
        ),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    prompt = _input_text(request)
    instructions = _instructions(request)
    assert "agent K" in prompt
    assert "agent L" in prompt
    assert "- `ACTION1`: up." in instructions
    assert "- `ACTION2`" not in instructions
    assert "- `ACTION3`" not in instructions
    assert request["text"]["format"]["schema"]["properties"]["action"][
        "properties"
    ]["action_id"]["enum"] == ["ACTION1"]


def test_openai_agent_can_include_output_schema_in_instructions() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(
            max_tool_calls=0,
            repair_attempts=0,
            include_output_schema_in_instructions=True,
        ),
        client=client,
    )

    adapter.decide(
        RoleContext(game="choose directly"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    instructions = _instructions(client.responses.calls[0])
    assert "Output JSON must match this schema exactly." in instructions
    assert '"action"' in instructions
    assert '"action_id"' in instructions
    assert '"ACTION1"' in instructions


def test_openai_agent_prompt_includes_recent_action_history() -> None:
    first, current = _observations()
    first_history_entry = ActionHistoryEntry(
        action=ActionSpec(
            action_id="ACTION6",
            data={"x": 32, "y": 20},
            target="the dark tile near the upper center",
        ),
        controllable=True,
        changed_pixel_count=3,
        change_summary="The coordinate action had no useful effect.",
        action_mode="probing",
    )
    latest_history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION2"),
        controllable=True,
        changed_pixel_count=7,
        change_summary="The cursor moved down.",
        action_mode="policy",
    )
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(general="agent K", game="agent L"),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(first_history_entry, latest_history_entry),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = _input_text(client.responses.calls[0])
    assert client.responses.calls[0]["instructions"]
    assert "agent K" in prompt
    assert "agent L" in prompt
    assert (
        "## Recent actions\n\n"
        "1. ACTION6 target=\"the dark tile near the upper center\" "
        "[mode=probing] [changed_pixels=3%] "
        "Elements and associated changes:\n"
        "The coordinate action had no useful effect.\n"
        "2. ACTION2 [latest] [mode=policy] [changed_pixels=7%] "
        "Elements and associated changes:\nThe cursor moved down."
    ) in prompt


def test_agent_prompt_includes_game_reset_history_marker() -> None:
    first, current = _observations()
    before_reset = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=2,
        change_summary="The agent moved.",
    )
    reset_marker = ActionHistoryResetMarker(
        reason="game_over_reset",
        restart_count=1,
    )
    after_reset = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION2"),
        controllable=True,
        changed_pixel_count=5,
        change_summary="The new game frame changed.",
    )
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(before_reset, reset_marker, after_reset),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = _input_text(client.responses.calls[0])
    assert (
        "1. ACTION1 [changed_pixels=2%] "
        "Elements and associated changes:\nThe agent moved.\n"
        "2. GAME_RESET [reason=game_over_reset] [restart_count=1]\n"
        "3. ACTION2 [latest] [changed_pixels=5%] "
        "Elements and associated changes:\nThe new game frame changed."
    ) in prompt


def test_openai_agent_prompt_marks_disabled_recent_actions_unavailable() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history_available=False,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = _input_text(client.responses.calls[0])
    assert "## Recent actions\n\nnot available" in prompt


def test_agent_prompt_marks_animation_history_entries() -> None:
    first, current = _observations()
    action_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=4,
        change_summary="The cursor moved right.",
    )
    animation_entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_count=0,
        change_summary="No visible change.",
    )
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(action_entry, animation_entry),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = _input_text(client.responses.calls[0])
    assert (
        "1. ACTION1 [latest] [animation] [changed_pixels=0%] "
        "Elements and associated changes:\n"
        "No changes happened for this transition. "
        "The previous and current frames are identical"
    ) in prompt
    assert "\n2. NONE [animation]" not in prompt
    assert "\n   - NONE [animation]" not in prompt


def test_agent_prompt_marks_orphan_animation_history_group() -> None:
    first, current = _observations()
    animation_entry = ActionHistoryEntry(
        action=ActionSpec.none(),
        controllable=False,
        changed_pixel_count=0,
        change_summary="No visible change.",
    )
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(animation_entry,),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = _input_text(client.responses.calls[0])
    assert (
        "1. animation_without_prior_action:\n"
        "   - [animation] [latest] [changed_pixels=0%] "
        "Elements and associated changes:\n"
        "No changes happened for this transition. "
        "The previous and current frames are identical"
    ) in prompt


def test_ollama_agent_prompt_includes_recent_action_history_changed_pixels() -> None:
    first, current = _observations()
    history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=5,
        change_summary="A tile flashed.",
    )
    client = FakeOllamaClient([_ollama_final_response()])
    adapter = OllamaOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        recent_action_history=(history_entry,),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    prompt = [
        message["content"]
        for message in client.calls[0]["messages"]
        if message["role"] == "user"
    ][-1]
    assert (
        "1. ACTION1 [latest] [changed_pixels=5%] "
        "Elements and associated changes:\nA tile flashed."
    ) in prompt
    assert client.calls[0]["messages"][-1] == {
        "role": "assistant",
        "content": "```json\n",
    }


def test_openai_agent_prompt_uses_current_image() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(
            available_tools=(),
        ),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    assert "current" in _instructions(request)
    assert len(_input_images(request)) == 1


def test_openai_agent_applies_configured_input_image_size_to_images() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(
        OpenAIOrchestratorAgentConfig(
            input_image_size="10x12",
            input_image_resample="nearest",
            max_tool_calls=0,
            repair_attempts=0,
        ),
        client=client,
    )

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    initial_images = _input_images(client.responses.calls[0])
    initial_sizes = [
        _decode_data_url_image(image["image_url"]).size
        for image in initial_images
    ]
    assert initial_sizes == [(10, 12)]


def test_openai_agent_uses_current_image_only() -> None:
    first, current = _observations()
    client = FakeOpenAIClient([_openai_final_response("resp-final")])
    adapter = OpenAIOrchestratorAgentAdapter(client=client)

    adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeRuntime(available_tools=()),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    request = client.responses.calls[0]
    images = _input_images(request)
    assert len(images) == 1
    assert [
        _decode_data_url_image(image["image_url"]).getpixel((0, 0))
        for image in images
    ] == [(255, 255, 255)]


def test_openai_agent_repairs_invalid_final_action_once() -> None:
    first, current = _observations()
    client = FakeOpenAIClient(
        [
            _openai_final_response("resp-invalid", action_id="BAD"),
            _openai_final_response("resp-valid", action_id="ACTION1"),
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
    repair_texts = [
        content["text"]
        for item in repair_input
        if isinstance(item, dict)
        for content in item.get("content", [])
        if content.get("type") == "input_text"
    ]
    assert any(text.startswith("Repair attempt 1:") for text in repair_texts)
    assert any("Invalid output:" in text for text in repair_texts)
    assert any('"action_id": "BAD"' in text for text in repair_texts)
    assert any(
        "Return only corrected final action JSON" in text for text in repair_texts
    )


def test_openai_tool_schema_requires_complex_action_coordinates() -> None:
    schema = final_action_schema([ActionSpec(action_id=GameAction.ACTION6)])

    action_schema = schema["properties"]["action"]
    data_schema = action_schema["properties"]["data"]

    assert action_schema["required"] == ["action_id", "data", "target"]
    assert data_schema["type"] == "object"
    assert data_schema["required"] == ["x", "y"]
    assert data_schema["properties"]["x"] == {"type": "number"}
    assert data_schema["properties"]["y"] == {"type": "number"}
    assert action_schema["properties"]["target"]["type"] == "string"


def test_final_action_schema_splits_mixed_simple_and_complex_actions() -> None:
    schema = final_action_schema(
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id=GameAction.ACTION6)]
    )

    simple_schema, complex_schema = schema["properties"]["action"]["anyOf"]

    assert simple_schema["properties"]["action_id"]["enum"] == ["ACTION1"]
    assert complex_schema["properties"]["action_id"]["enum"] == ["ACTION6"]
    assert complex_schema["required"] == ["action_id", "data", "target"]
    assert complex_schema["properties"]["data"]["required"] == ["x", "y"]


def test_parse_action_scales_profiled_normalized_coordinates() -> None:
    action = parse_action(
        {
            "action_id": "ACTION6",
            "data": {"x": 500, "y": 1000},
            "target": "the lower middle object",
        },
        [ActionSpec(action_id=GameAction.ACTION6)],
        coordinate_space="normalized_1000",
    )

    assert action.data == {"x": 32, "y": 63}
    assert action.target == "the lower middle object"


def test_parse_action_scales_normalized_coordinates_through_crop() -> None:
    action = parse_action(
        {
            "action_id": "ACTION6",
            "data": {"x": 0, "y": 1000},
            "target": "the lower left corner",
        },
        [ActionSpec(action_id=GameAction.ACTION6)],
        coordinate_space="normalized_1000",
        arc_grid_crop_edges=(4, 4, 4, 4),
    )

    assert action.data == {"x": 4, "y": 59}
    assert action.target == "the lower left corner"


def test_parse_action_rejects_action6_without_target() -> None:
    with pytest.raises(Exception, match="ACTION6 requires non-empty action.target"):
        parse_action(
            {
                "action_id": "ACTION6",
                "data": {"x": 500, "y": 1000},
            },
            [ActionSpec(action_id=GameAction.ACTION6)],
            coordinate_space="normalized_1000",
        )


def test_parse_action_accepts_simple_action_without_data() -> None:
    action = parse_action(
        {
            "action_id": "ACTION1",
        },
        [ActionSpec(action_id="ACTION1")],
    )

    assert action.data is None


def test_parse_action_rejects_simple_action_with_coordinate_data() -> None:
    with pytest.raises(Exception, match="must not include action.data"):
        parse_action(
            {
                "action_id": "ACTION1",
                "data": {"x": 1, "y": 2},
            },
            [ActionSpec(action_id="ACTION1")],
        )


def test_parse_action_rejects_out_of_profile_coordinates() -> None:
    with pytest.raises(Exception, match="normalized 0..1000"):
        parse_action(
            {
                "action_id": "ACTION6",
                "data": {"x": 1001, "y": 0},
                "target": "the upper right object",
            },
            [ActionSpec(action_id=GameAction.ACTION6)],
            coordinate_space="normalized_1000",
        )


def test_ollama_agent_repairs_invalid_final_action_once() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            _ollama_final_response(action_id="BAD"),
            _ollama_final_response(),
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
    repair_messages = [
        message.get("content", "")
        for message in client.calls[1]["messages"]
        if message.get("role") == "user"
    ]
    assert any(text.startswith("Repair attempt 1:") for text in repair_messages)
    assert any("Invalid output:" in text for text in repair_messages)
    assert any('"action_id": "BAD"' in text for text in repair_messages)


def test_ollama_agent_rejects_plain_text_final_action() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": (
                        "The previous action was ACTION1. I will choose "
                        "ACTION2 to probe below."
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

    with pytest.raises(RuntimeError, match="invalid final structured action"):
        adapter.decide(
            RoleContext(),
            current,
            [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")],
            FakeRuntime(available_tools=()),
            first_observation_ref=ObservationRef(memory="state", id=first.id),
        )


def test_ollama_agent_rejects_unstructured_action_alias_text() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "I will keep moving right to probe the path.",
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

    with pytest.raises(RuntimeError, match="invalid final structured action"):
        adapter.decide(
            RoleContext(
                game=(
                    "Action mapping notes:\n"
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


def test_ollama_agent_uses_structured_final_action_format() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            _ollama_final_response(
                action_id="ACTION2",
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
    assert client.calls[0]["format"] == final_action_schema(
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")]
    )
    system_content = client.calls[0]["messages"][0]["content"]
    assert "Output contract:" not in system_content
    assert "## Output JSON" in system_content
    assert "`action.action_id`" in system_content
    assert client.calls[0]["messages"][-2]["role"] == "user"
    assert client.calls[0]["messages"][-1] == {
        "role": "assistant",
        "content": "```json\n",
    }


def test_ollama_agent_two_passes_when_thinking_enabled() -> None:
    first, current = _observations()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "I will take ACTION2.",
                    "thinking": "looked for a better direction",
                    "tool_calls": [],
                },
                done_reason="stop",
            ),
            _ollama_final_response(action_id="ACTION2"),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(
            think=True,
            max_tool_calls=0,
            repair_attempts=0,
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
    assert len(client.calls) == 2
    assert "format" not in client.calls[0]
    assert client.calls[0]["think"] is True
    assert client.calls[0]["messages"][-1]["role"] == "user"
    assert client.calls[1]["format"] == final_action_schema(
        [ActionSpec(action_id="ACTION1"), ActionSpec(action_id="ACTION2")]
    )
    assert client.calls[1]["think"] is False
    assert client.calls[1]["messages"][-1] == {
        "role": "assistant",
        "content": "```json\n",
    }
    records = drain_model_input_debug_records(adapter)
    assert [record["phase"] for record in records] == [
        "final_action_thinking",
        "final_action",
    ]
    assert records[0]["metadata"]["response_payload"]["message"]["thinking"] == (
        "looked for a better direction"
    )


def test_ollama_agent_does_not_prefill_when_tools_are_exposed() -> None:
    first, current = _observations()
    client = FakeOllamaClient([_ollama_final_response()])
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(max_tool_calls=1, repair_attempts=0),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        FakeGenericRuntime(),
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert client.calls[0]["messages"][-1]["role"] == "user"
    assert client.calls[0]["tools"][0]["function"]["name"] == "inspect"


def test_ollama_agent_thinking_tool_call_skips_json_conversion() -> None:
    first, current = _observations()
    runtime = FakeGenericRuntime()
    client = FakeOllamaClient(
        [
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "",
                    "thinking": "inspect before deciding",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "inspect",
                                "arguments": {"source_state_id": 3},
                            }
                        }
                    ],
                },
                done_reason="stop",
            ),
            SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": "I can now answer ACTION1.",
                    "thinking": "used the inspection result",
                    "tool_calls": [],
                },
                done_reason="stop",
            ),
            _ollama_final_response(action_id="ACTION1"),
        ]
    )
    adapter = OllamaOrchestratorAgentAdapter(
        OllamaOrchestratorAgentConfig(
            think=True,
            max_tool_calls=1,
            repair_attempts=0,
        ),
        client=client,
    )

    decision = adapter.decide(
        RoleContext(),
        current,
        [ActionSpec(action_id="ACTION1")],
        runtime,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    assert decision.final_action.action_id == "ACTION1"
    assert [call.tool for call in runtime.calls] == ["inspect"]
    assert len(client.calls) == 3
    assert "format" not in client.calls[0]
    assert "format" not in client.calls[1]
    assert client.calls[2]["format"] == final_action_schema(
        [ActionSpec(action_id="ACTION1")]
    )
    records = drain_model_input_debug_records(adapter)
    assert [record["phase"] for record in records] == [
        "final_action_thinking",
        "final_action_thinking",
        "final_action",
    ]
