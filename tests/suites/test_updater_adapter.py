"""Tests for vLLM updater prompts."""

from __future__ import annotations

import json
from typing import Any

from arcengine import GameAction
import pytest

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.updater import (
    AGENT_GAME_CONTEXT_KEYS,
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdateResult,
    PromptUpdaterAdapter,
    UpdaterConfig,
    UpdaterContextTarget,
    UpdaterOutputError,
    agent_game_updated_context_json_schema,
    parse_agent_game_updated_context_output,
    parse_updated_context_output,
    updated_context_json_schema,
)
from face_of_agi.models.updater.config import VLLMUpdaterConfig
from face_of_agi.models.updater.providers.vllm import (
    VLLMUpdaterAdapter,
    VLLMUpdaterProvider,
)


class FakePromptUpdaterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[PromptUpdateRequest] = []

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        if request.target.task == "agent_game":
            payload = {
                "updated_context": {
                    key: f"{request.target.role}-{key}"
                    for key in AGENT_GAME_CONTEXT_KEYS
                }
            }
        else:
            payload = {"updated_context": f"{request.target.role}-general"}
        return PromptUpdateProviderResponse(
            target=request.target,
            text=json.dumps(payload),
        )

    def repair_prompt(self, *args: Any, **kwargs: Any) -> PromptUpdateProviderResponse:
        raise AssertionError("repair should not be needed")


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


def _agent_update_input() -> AgentGameContextUpdateInput:
    grid = _grid()
    grid[9][8] = 3
    return AgentGameContextUpdateInput(
        previous_context=RoleContext(general="general", game="old game context"),
        current_observation=Observation(id="obs-1", step=1, frame=grid),
        allowed_actions=(ActionSpec("ACTION1"), ActionSpec(GameAction.ACTION6)),
        glossary_actions=(ActionSpec("ACTION1"), ActionSpec(GameAction.ACTION6)),
        action_history_window=3,
    )


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "id": "resp-updater",
        "model": "fake-vllm",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
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


def test_prompt_updater_builds_agent_observation_text_request() -> None:
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(provider=provider, config=UpdaterConfig())

    result = updater.update_agent_game_context(_agent_update_input())

    assert "agent-goals" in result.game
    request = provider.requests[0]
    assert request.target.role == "agent"
    assert request.target.task == "agent_game"
    _assert_no_stale_text_prompt_terms(request.instructions)
    _assert_no_stale_text_prompt_terms(request.text)
    assert "ARC grid 0..63 coordinates" not in request.instructions
    assert "visible cropped coordinates" in request.instructions
    assert "3 to 60" in request.instructions
    assert "shape, colors" not in request.instructions
    assert "ARC color glossary" in request.instructions
    assert "0=white" not in request.instructions
    assert "A=cyan" not in request.instructions
    assert "symbol A (cyan)" not in request.instructions
    assert "canonical glossary colors" not in request.instructions
    assert "symbol A: light cyan" in request.instructions
    assert "symbol 0" in request.instructions
    assert "symbol F" in request.instructions
    assert "A-cells" in request.instructions
    assert "Action history window" not in request.instructions
    assert len(request.images) == 1
    assert request.images[0].label == "current_observation"
    assert request.images[0].image.size == (2048, 2048)
    assert "## Current observation" in request.text
    assert "## current_observation\n\n### frame 0" in request.text
    assert "x_range: 3..60" in request.text
    assert "ACTION6(x,y 3..60,target)" in request.text
    assert "## Action history window" not in request.text
    assert "observation_id:" not in request.text
    assert "crop_bounds_original_xyxy:" not in request.text
    assert "coordinate_system:" not in request.text
    assert "symbols:" not in request.text
    assert "ARC color symbols" not in request.text
    assert "image_url" not in request.text
    assert "base64" not in request.text


def test_vllm_updater_sends_plain_text_message() -> None:
    updated_context = {
        "updated_context": {
            key: f"updated {key}" for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    client = FakeClient([_chat_response(json.dumps(updated_context))])
    updater = VLLMUpdaterAdapter(
        VLLMUpdaterConfig(model="fake-vllm", repair_attempts=0),
        client=client,
    )

    result = updater.update_agent_game_context(_agent_update_input())

    assert "updated goals" in result.game
    request = client.chat.completions.calls[0]
    _assert_no_stale_text_prompt_terms(json.dumps(request))
    text = _text_part(request["messages"][1]["content"])
    assert "## Current observation" in text
    image_parts = _image_parts(request["messages"][1]["content"])
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    serialized = json.dumps(request)
    assert "image_url" in serialized
    assert "data:image/png;base64," in serialized


def test_general_updater_uses_text_json_payload() -> None:
    provider = FakePromptUpdaterProvider()
    updater = PromptUpdaterAdapter(provider=provider, config=UpdaterConfig())

    result = updater.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="agent",
            previous_context=RoleContext(general="old general", game="game"),
            run_id="run-1",
            game_id="game-1",
        )
    )

    assert result.general == "agent-general"
    request = provider.requests[0]
    assert request.target.task == "general"
    assert isinstance(json.loads(request.text), dict)
    assert "ARC color glossary" not in request.instructions


def test_updater_output_parsers_accept_expected_json() -> None:
    assert parse_updated_context_output('{"updated_context": "new"}') == "new"

    parsed = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "updated_context": {
                    key: f"value {key}" for key in AGENT_GAME_CONTEXT_KEYS
                }
            }
        )
    )
    assert "value goals" in parsed


def test_updater_schemas_include_configured_max_lengths() -> None:
    general_schema = updated_context_json_schema(general_context_max_chars=123)
    agent_schema = agent_game_updated_context_json_schema(
        agent_game_context_max_chars=456,
        agent_game_context_field_max_chars=78,
    )

    assert general_schema["properties"]["updated_context"]["maxLength"] == 123
    updated_context_schema = agent_schema["properties"]["updated_context"]
    assert "456 characters" in updated_context_schema["description"]
    assert (
        updated_context_schema["properties"]["goals"]["maxLength"]
        == 78
    )


def test_updater_parsers_reject_oversized_output() -> None:
    with pytest.raises(UpdaterOutputError, match="too long"):
        parse_updated_context_output(
            '{"updated_context": "abcdef"}',
            max_chars=5,
        )

    payload = {
        "updated_context": {
            key: ("abcdef" if key == "goals" else f"value {key}")
            for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    with pytest.raises(UpdaterOutputError, match="too long"):
        parse_agent_game_updated_context_output(
            json.dumps(payload),
            field_max_chars=5,
        )


def test_vllm_updater_provider_clips_invalid_output_in_repair_prompt() -> None:
    client = FakeClient([_chat_response('{"updated_context": "ok"}')])
    provider = VLLMUpdaterProvider(
        VLLMUpdaterConfig(
            model="fake-vllm",
            repair_invalid_output_preview_chars=80,
        ),
        client=client,
    )
    target = UpdaterContextTarget(
        role="agent",
        segment="general",
        task="general",
        previous_context=RoleContext(general="old", game="game"),
    )
    request = PromptUpdateRequest(
        target=target,
        instructions="instructions",
        text="input",
        output_schema=updated_context_json_schema(),
    )
    invalid_text = "a" * 120 + "TAIL"

    provider.repair_prompt(
        request,
        invalid_text=invalid_text,
        validation_error="bad",
        attempt=1,
    )

    repair_text = client.chat.completions.calls[0]["messages"][1]["content"]
    assert isinstance(repair_text, str)
    assert "Invalid output preview:" in repair_text
    assert "omitted" in repair_text
    assert "TAIL" in repair_text
    assert invalid_text not in repair_text
