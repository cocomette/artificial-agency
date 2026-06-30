"""Tests for the vLLM orchestrator-agent adapter."""

from __future__ import annotations

import json
from typing import Any

from arcengine import GameAction
import pytest

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.observation_text import ObservationTextConfig
from face_of_agi.models.orchestrator_agent import VLLMOrchestratorAgentConfig
from face_of_agi.models.orchestrator_agent.providers.vllm import (
    VLLMOrchestratorAgentAdapter,
)


class FakeCompletions:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeCompletions(responses)},
        )()


def _grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "id": "resp-1",
        "model": "fake-vllm",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12},
    }


def _assert_no_stale_text_prompt_terms(text: str) -> None:
    lower_text = text.lower()
    stale_terms = (
        "attached image",
        "attached frame",
        "current image frame",
        "0..1000",
        "0 to 1000",
    )
    for term in stale_terms:
        assert term not in lower_text


def _text_part(content: Any) -> str:
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    return content[0]["text"]


def _image_parts(content: Any) -> list[dict[str, Any]]:
    assert isinstance(content, list)
    return [part for part in content if part.get("type") == "image_url"]


def test_vllm_agent_sends_text_only_observation_prompt_and_parses_action6() -> None:
    grid = _grid()
    grid[12][13] = 4
    client = FakeClient(
        [
            _chat_response(
                json.dumps(
                    {
                        "action": {
                            "action_id": "ACTION6",
                            "data": {"x": 13, "y": 12},
                            "target": "symbol 4 cell",
                        }
                    }
                )
            )
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(model="fake-vllm", repair_attempts=0),
        client=client,
    )

    decision = adapter.decide(
        context=RoleContext(game="try the marked cell"),
        current_observation=Observation(id="obs-1", step=3, frame=grid),
        action_space=(ActionSpec(GameAction.ACTION6),),
        glossary_actions=(ActionSpec(GameAction.ACTION6),),
    )

    assert decision.final_action.name == "ACTION6"
    assert decision.final_action.data == {"x": 13, "y": 12}
    assert decision.final_action.target == "symbol 4 cell"
    request = client.chat.completions.calls[0]
    assert request["model"] == "fake-vllm"
    instructions = request["messages"][0]["content"]
    _assert_no_stale_text_prompt_terms(instructions)
    assert '"x":<0..63>' not in instructions
    assert '"y":<0..63>' not in instructions
    assert '"x":<visible-crop-x>' in instructions
    assert "3 to 60" in instructions
    assert "visible cropped coordinates" in instructions
    assert "ARC color glossary" in instructions
    assert "0=white" not in instructions
    assert "A=cyan" not in instructions
    assert "symbol A (cyan)" not in instructions
    assert "canonical glossary colors" not in instructions
    assert "symbol A: light cyan" in instructions
    user_prompt = _text_part(request["messages"][1]["content"])
    assert "## Current observation" in user_prompt
    image_parts = _image_parts(request["messages"][1]["content"])
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_parts[0]["image_url"]["detail"] == "auto"
    assert "## current_observation\n\n### frame 0" in user_prompt
    assert "x_range: 3..60" in user_prompt
    assert "ACTION6(x,y 3..60,target)" in user_prompt
    assert "observation_id:" not in user_prompt
    assert "crop_bounds_original_xyxy:" not in user_prompt
    assert "coordinate_system:" not in user_prompt
    assert "symbols:" not in user_prompt
    assert "ARC color symbols" not in user_prompt
    assert "image_url" in json.dumps(request)
    assert "data:image/png;base64," in json.dumps(request)
    serialized_request = json.dumps(request)
    assert '"minimum": 3' in serialized_request
    assert '"maximum": 60' in serialized_request


def test_vllm_agent_action6_validation_uses_configured_visible_crop() -> None:
    grid = _grid()
    client = FakeClient(
        [
            _chat_response(
                json.dumps(
                    {
                        "action": {
                            "action_id": "ACTION6",
                            "data": {"x": 2, "y": 61},
                            "target": "crop edge",
                        }
                    }
                )
            )
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model="fake-vllm",
            repair_attempts=0,
            observation_text=ObservationTextConfig(crop_cells=2),
        ),
        client=client,
    )

    decision = adapter.decide(
        context=RoleContext(game="try the crop edge"),
        current_observation=Observation(id="obs-1", step=3, frame=grid),
        action_space=(ActionSpec(GameAction.ACTION6),),
        glossary_actions=(ActionSpec(GameAction.ACTION6),),
    )

    assert decision.final_action.data == {"x": 2, "y": 61}
    assert decision.final_action.target == "crop edge"
    request = client.chat.completions.calls[0]
    instructions = request["messages"][0]["content"]
    user_prompt = _text_part(request["messages"][1]["content"])
    assert "2 to 61" in instructions
    assert "ACTION6(x,y 2..61,target)" in user_prompt
    serialized_request = json.dumps(request)
    assert '"minimum": 2' in serialized_request
    assert '"maximum": 61' in serialized_request


def test_vllm_agent_rejects_action6_outside_configured_visible_crop() -> None:
    grid = _grid()
    client = FakeClient(
        [
            _chat_response(
                json.dumps(
                    {
                        "action": {
                            "action_id": "ACTION6",
                            "data": {"x": 1, "y": 61},
                            "target": "outside crop",
                        }
                    }
                )
            )
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(
            model="fake-vllm",
            repair_attempts=0,
            observation_text=ObservationTextConfig(crop_cells=2),
        ),
        client=client,
    )

    with pytest.raises(RuntimeError, match="visible crop 2..61"):
        adapter.decide(
            context=RoleContext(game="try the crop edge"),
            current_observation=Observation(id="obs-1", step=3, frame=grid),
            action_space=(ActionSpec(GameAction.ACTION6),),
            glossary_actions=(ActionSpec(GameAction.ACTION6),),
        )


def test_vllm_agent_requires_action6_target() -> None:
    grid = _grid()
    client = FakeClient(
        [
            _chat_response(
                json.dumps(
                    {
                        "action": {
                            "action_id": "ACTION6",
                            "data": {"x": 13, "y": 12},
                        }
                    }
                )
            )
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(model="fake-vllm", repair_attempts=0),
        client=client,
    )

    with pytest.raises(RuntimeError, match="action.target"):
        adapter.decide(
            context=RoleContext(game="try the marked cell"),
            current_observation=Observation(id="obs-1", step=3, frame=grid),
            action_space=(ActionSpec(GameAction.ACTION6),),
            glossary_actions=(ActionSpec(GameAction.ACTION6),),
        )


def test_vllm_agent_rejects_simple_action_target() -> None:
    grid = _grid()
    client = FakeClient(
        [
            _chat_response(
                json.dumps(
                    {
                        "action": {
                            "action_id": "ACTION1",
                            "target": "not allowed",
                        }
                    }
                )
            )
        ]
    )
    adapter = VLLMOrchestratorAgentAdapter(
        VLLMOrchestratorAgentConfig(model="fake-vllm", repair_attempts=0),
        client=client,
    )

    with pytest.raises(RuntimeError, match="simple actions must not include"):
        adapter.decide(
            context=RoleContext(game="move"),
            current_observation=Observation(id="obs-1", step=3, frame=grid),
            action_space=(ActionSpec(GameAction.ACTION1),),
            glossary_actions=(ActionSpec(GameAction.ACTION1),),
        )


def test_vllm_agent_rejects_tool_calls() -> None:
    config = VLLMOrchestratorAgentConfig(model="fake-vllm", max_tool_calls=1)

    try:
        VLLMOrchestratorAgentAdapter(config, client=FakeClient([]))
    except ValueError as exc:
        assert "does not support tool calls" in str(exc)
    else:
        raise AssertionError("expected max_tool_calls rejection")
