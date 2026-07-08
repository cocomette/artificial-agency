"""Tests for the active updater P prompt adapter task."""

from __future__ import annotations

import json
from typing import Any

import pytest
from arcengine import GameAction
from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    UpdaterConfig,
    UpdaterOutputError,
    agent_game_updated_context_json_schema,
    parse_agent_game_updated_context_output,
    updater_instruction_path,
)
from face_of_agi.models.change.components import arc_rendered_color_map


class FakeUpdaterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[PromptUpdateRequest] = []

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return PromptUpdateProviderResponse(
            target=request.target,
            text=response,
        )

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        return PromptUpdateProviderResponse(
            target=request.target,
            text=self.responses.pop(0),
            metadata={
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            },
        )


def test_instruction_path_uses_single_agent_task(tmp_path) -> None:
    assert updater_instruction_path(
        task="agent",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_context_updater_prompt.md"


def test_agent_updater_returns_structured_agent_context(tmp_path) -> None:
    _write_instruction_file(tmp_path)
    response = {
        "current_strategy": "reach the red door",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="newest strategy"),
            previous_game_context_history=("older strategy",),
            world_model_context="world_description: world updated",
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    request = provider.requests[0]
    assert request.target.role == "agent"
    assert request.target.task == "agent"
    assert request.output_schema == agent_game_updated_context_json_schema(
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
    )
    assert "current_strategy" in request.output_schema["properties"]
    assert "## Action glossary" in request.instructions
    assert request.images[0].label == "current_observation_frame"
    assert json.loads(result.context) == {
        "current_strategy": "reach the red door",
    }
    assert result.next_actions[0].name == "ACTION1"
    assert "## Previous current_strategy" in request.text
    assert "## Previous game context\n" not in request.text
    assert "older strategy" in request.text
    assert "newest strategy" in request.text
    assert "## Action history" in request.text
    assert "## Previous strategy summary" in request.text
    assert "## Previous actions summary" in request.text
    assert "## World model" in request.text
    assert "world updated" in request.text


def test_agent_updater_can_include_current_frame_components(tmp_path) -> None:
    _write_instruction_file(tmp_path)
    response = {
        "current_strategy": "reach the purple block",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(
            instruction_dir=str(tmp_path),
            input_image_crop_arc_grid_edges=None,
        ),
    )
    frame = _arc_grid()
    for y in range(20, 22):
        for x in range(10, 12):
            frame[y][x] = 4

    updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old strategy"),
            current_observation=Observation(id="current", step=1, frame=frame),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    request = provider.requests[0]
    text = request.text
    assert "## Current frame components" in text
    assert "frame 0:" not in text
    assert "- color=charcoal nb=1 box=[(156,312,188,344)]" in text
    assert "symbol=" not in text
    assert "rgb=" not in text
    assert "## ARC rendered color legend" not in text
    assert "## ARC rendered color legend" not in request.instructions
    assert text.index("## Allowed actions") < text.index(
        "## Current frame components"
    )
    assert text.index("## Current frame components") < text.index(
        "## Previous strategy summary"
    )


def test_agent_updater_retargets_action6_by_color_then_bbox_center(tmp_path) -> None:
    _write_instruction_file(tmp_path)
    response = {
        "current_strategy": "target the red tile",
        "next_actions": [
            {
                "action_id": "ACTION6",
                "target": "the red tile",
                "bbox": [312, 312, 469, 469],
                "target_rgb_color": list(arc_rendered_color_map()[2]),
            }
        ],
    }
    frame = _arc_grid(0)
    frame[24][24] = 1
    frame[21][21] = 2
    frame[25][27] = 2
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(
            instruction_dir=str(tmp_path),
            input_image_crop_arc_grid_edges=None,
        ),
    )

    result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=Observation(id="obs-64", step=1, frame=frame),
            allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
            glossary_actions=(ActionSpec(action_id=GameAction.ACTION6),),
        )
    )

    assert provider.requests[0].images[0].image.size == (64, 64)
    assert result.next_actions[0].data == {"x": 27, "y": 25}
    assert result.next_actions[0].target == "the red tile"
    assert result.next_actions[0].target_value == 2
    assert result.next_actions[0].target_bbox == (312, 312, 469, 469)


def test_agent_updater_prompt_includes_action_history(tmp_path) -> None:
    _write_instruction_file(tmp_path)
    response = {
        "current_strategy": "choose action one",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(provider, UpdaterConfig(instruction_dir=str(tmp_path)))

    updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation(),
            allowed_actions=(
                ActionSpec(action_id="ACTION1"),
                ActionSpec(action_id=GameAction.ACTION6),
            ),
            glossary_actions=(
                ActionSpec(action_id="ACTION1"),
                ActionSpec(action_id=GameAction.ACTION6),
            ),
            action_history=(
                ActionHistoryEntry(
                    action=ActionSpec(action_id="ACTION1"),
                    controllable=True,
                    changed_pixel_count=4,
                    change_summary=(
                        "- target_tile: red tile in the lower middle; "
                        "mutations: moved right"
                    ),
                ),
            ),
        )
    )

    text = provider.requests[0].text
    assert "## Action history" in text
    assert "ACTION1" in text
    assert "[changed_pixels=4%]" in text
    assert "target_tile" in text
    assert "moved right" in text


def test_agent_updater_repairs_invalid_output(tmp_path) -> None:
    _write_instruction_file(tmp_path)
    repaired = {
        "current_strategy": "repair plan",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider("{}", json.dumps(repaired))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=1),
    )

    result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )

    assert len(provider.requests) == 2
    assert json.loads(result.context) == {
        "current_strategy": "repair plan",
    }


def test_agent_updater_falls_back_after_repair_exhaustion(tmp_path, caplog) -> None:
    _write_instruction_file(tmp_path)
    provider = FakeUpdaterProvider("{}")
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=0),
    )

    with caplog.at_level("WARNING"):
        result = updater.update_agent_context(
            AgentGameContextUpdateInput(
                previous_context=RoleContext(
                    game=json.dumps(
                        {
                            "current_strategy": "keep current plan",
                        }
                    )
                ),
                current_observation=Observation(
                    id="obs-64",
                    step=1,
                    frame=_arc_grid(0),
                ),
                allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
                glossary_actions=(ActionSpec(action_id=GameAction.ACTION6),),
                actions_window=2,
            )
        )

    assert json.loads(result.context) == {
        "current_strategy": "keep current plan",
    }
    assert len(result.next_actions) == 2
    assert result.next_actions[0].name == "ACTION6"
    assert result.next_actions[0].data == {"x": 32, "y": 32}
    assert result.next_actions[0].target_value == 0
    assert (
        "max repair attempts / model context length reached, continuing with "
        "previous-context fallback"
    ) in caplog.text
    assert "Traceback" not in caplog.text


def test_agent_updater_falls_back_after_provider_context_length(
    tmp_path,
    caplog,
) -> None:
    _write_instruction_file(tmp_path)
    provider = FakeUpdaterProvider(RuntimeError("maximum context length reached"))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=1),
    )

    with caplog.at_level("WARNING"):
        result = updater.update_agent_context(
            AgentGameContextUpdateInput(
                previous_context=RoleContext(
                    game=json.dumps(
                        {
                            "current_strategy": "keep current plan",
                        }
                    )
                ),
                current_observation=Observation(
                    id="obs-64",
                    step=1,
                    frame=_arc_grid(0),
                ),
                allowed_actions=(ActionSpec(action_id="ACTION1"),),
                glossary_actions=(ActionSpec(action_id="ACTION1"),),
            )
        )

    assert json.loads(result.context) == {
        "current_strategy": "keep current plan",
    }
    assert len(result.next_actions) == 1
    assert result.next_actions[0].name == "ACTION1"
    assert "RuntimeError: maximum context length reached" in caplog.text
    assert "Traceback" not in caplog.text


def test_agent_game_schema_uses_next_actions_window() -> None:
    schema = agent_game_updated_context_json_schema(
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
        actions_window=3,
    )

    assert schema["required"] == [
        "current_strategy",
        "next_actions",
    ]
    next_actions_schema = schema["properties"]["next_actions"]
    assert next_actions_schema["minItems"] == 3
    assert next_actions_schema["maxItems"] == 3


def test_agent_game_schema_requires_action6_target_bbox_and_rgb() -> None:
    schema = agent_game_updated_context_json_schema(
        allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
    )

    action_schema = schema["properties"]["next_actions"]["items"]
    assert action_schema["required"] == [
        "action_id",
        "target",
        "bbox",
        "target_rgb_color",
    ]


def test_agent_game_parser_accepts_context_and_next_actions() -> None:
    context, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "current_strategy": "plan",
                "next_actions": [{"action_id": "ACTION1"}],
            }
        ),
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert json.loads(context) == {
        "current_strategy": "plan",
    }
    assert next_actions[0].name == "ACTION1"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"next_actions": [{"action_id": "ACTION1"}]}, "missing keys"),
        ({"current_strategy": "plan", "next_actions": []}, "must not be empty"),
        (
            {
                "current_strategy": "plan",
                "next_actions": [{"action_id": "ACTION1"}],
            },
            "expected exactly the 2 action window",
        ),
        (
            {
                "current_strategy": "plan",
                "next_actions": [{"action_id": "ACTION2"}],
            },
            "invalid",
        ),
        (
            {
                "current_strategy": "plan",
                "next_actions": [{"action_id": "ACTION1"}],
                "extra": "nope",
            },
            "unexpected keys",
        ),
    ],
)
def test_agent_game_parser_rejects_invalid_next_actions(
    payload: dict[str, Any],
    match: str,
) -> None:
    actions_window = 2 if match.startswith("expected") else 1
    with pytest.raises(UpdaterOutputError, match=match):
        parse_agent_game_updated_context_output(
            json.dumps(payload),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            actions_window=actions_window,
        )


def test_agent_game_parser_accepts_action6_bbox_and_rgb_targeting() -> None:
    _, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "current_strategy": "probe center",
                "next_actions": [
                    {
                        "action_id": "ACTION6",
                        "target": "the lower left tile",
                        "bbox": [0, 800, 200, 1000],
                        "target_rgb_color": [10, 20, 30],
                    }
                ],
            }
        ),
        allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
        arc_grid_crop_edges=(4, 4, 4, 4),
    )

    assert next_actions[0].data == {
        "bbox": [0, 800, 200, 1000],
        "target_rgb_color": [10, 20, 30],
    }
    assert next_actions[0].target == "the lower left tile"


def _write_instruction_file(path) -> None:
    (path / "agent_context_updater_prompt.md").write_text(
        "agent instructions",
        encoding="utf-8",
    )


def _observation() -> Observation:
    return Observation(
        id="obs-1",
        step=1,
        frame=_arc_grid(),
    )


def _arc_grid(fill: int = 0) -> list[list[int]]:
    return [[fill for _x in range(64)] for _y in range(64)]
