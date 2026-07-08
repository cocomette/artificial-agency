"""Tests for the active updater P prompt adapter tasks."""

from __future__ import annotations

import json
from typing import Any

import pytest
from arcengine import GameAction
from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    RoleContext,
    SamePastStateDetection,
)
from face_of_agi.models.historizer import AgentContextHistorySummary
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    GeneralKnowledgeUpdateInput,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    UpdaterConfig,
    UpdaterOutputError,
    agent_game_updated_context_json_schema,
    parse_agent_game_updated_context_output,
    updated_context_json_schema,
    updater_instruction_path,
)


class FakeUpdaterProvider:
    backend = "fake"
    model = "fake-model"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests: list[PromptUpdateRequest] = []

    def update_prompt(
        self,
        request: PromptUpdateRequest,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        text = self.responses.pop(0)
        return PromptUpdateProviderResponse(target=request.target, text=text)

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        self.requests.append(request)
        text = self.responses.pop(0)
        return PromptUpdateProviderResponse(
            target=request.target,
            text=text,
            metadata={
                "invalid_text": invalid_text,
                "validation_error": validation_error,
                "attempt": attempt,
            },
        )


def test_instruction_paths_cover_active_tasks(tmp_path) -> None:
    assert updater_instruction_path(
        task="agent_probing",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_probing_context_updater_prompt.md"
    assert updater_instruction_path(
        task="agent_policy",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_policy_context_updater_prompt.md"
    assert updater_instruction_path(
        task="general",
        role="agent",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_general_context_updater_prompt.md"


def test_agent_game_updater_returns_structured_agent_context(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "probing_strategy": "try the newly opened path",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_agent_probing_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            context_history=_history_summary("world updated"),
        )
    )

    request = provider.requests[0]
    assert request.target.role == "agent"
    assert request.target.task == "agent_probing"
    assert request.output_schema == agent_game_updated_context_json_schema(
        mode="probing",
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
    )
    assert "probing_strategy" in request.output_schema["properties"]
    assert "## Action glossary" in request.instructions
    assert "- `ACTION1`: up." in request.instructions
    assert "- `ACTION2`" not in request.instructions
    assert "`Same past state detected`" in request.instructions
    assert "This input intentionally contains only strategy fields" in (
        request.instructions
    )
    assert request.images[0].label == "current_observation_frame"
    assert json.loads(result.context) == {
        "probing_strategy": "try the newly opened path"
    }
    assert result.next_actions[0].name == "ACTION1"
    assert result.updater_mode == "probing"
    assert "## Previous game context" in request.text
    assert "## World model" in request.text
    assert "world_description: world updated" in request.text
    assert "special_events: none" in request.text
    assert "action_effects:" in request.text
    assert "world updated" in request.text
    assert "## Probing evolution" in request.text
    assert "probing evolved" in request.text
    assert "## Policy evolution" in request.text
    assert "policy evolved" in request.text


def test_agent_policy_updater_prompt_includes_historizer_evolution(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "policy_strategy": "reach the exit",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_agent_policy_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(
                game=json.dumps(
                    {
                        "probing_strategy": "probe actions",
                        "policy_strategy": "reach target",
                    }
                )
            ),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            context_history=_history_summary("world updated"),
        )
    )

    request = provider.requests[0]
    assert request.target.task == "agent_policy"
    assert json.loads(result.context) == {"policy_strategy": "reach the exit"}
    assert "## World model" in request.text
    assert "world_description: world updated" in request.text
    assert "action_effects:" in request.text
    assert "## Probing evolution" in request.text
    assert "probing evolved" in request.text
    assert "## Policy evolution" in request.text
    assert "policy evolved" in request.text


def test_agent_game_updater_prompt_includes_same_past_state_detection(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "probing_strategy": "try a new route",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    updater.update_agent_probing_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            context_history=_history_summary("world updated"),
            same_past_state_detections=(
                SamePastStateDetection(
                    probing_strategy="probe the old loop",
                    policy_strategy="follow the old loop",
                    probing_evolution="probing repeated itself",
                    policy_evolution="policy repeated itself",
                ),
            ),
        )
    )

    text = provider.requests[0].text
    assert "## Same past state detected" in text
    assert "you were in this exact state before" in text
    assert "probe the old loop" in text
    assert "policy repeated itself" in text
    assert "direct next action" not in text
    assert "state_id=" not in text
    assert "turn_id=" not in text
    assert "action:" not in text


def test_agent_game_updater_retargets_action6_by_color_then_bbox_center(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "probing_strategy": "target the red tile",
        "next_actions": [
            {
                "action_id": "ACTION6",
                "target": "the red tile",
                "bbox": [312, 312, 469, 469],
                "target_rgb_color": [255, 0, 0],
            }
        ],
    }
    frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    frame.putpixel((24, 24), (0, 0, 255))
    frame.putpixel((21, 21), (255, 0, 0))
    frame.putpixel((27, 25), (255, 0, 0))
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(
            instruction_dir=str(tmp_path),
            input_image_crop_arc_grid_edges=None,
        ),
    )

    result = updater.update_agent_probing_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=Observation(
                id="obs-64",
                step=1,
                frame=frame,
            ),
            allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
            glossary_actions=(ActionSpec(action_id=GameAction.ACTION6),),
            context_history=_history_summary("world updated"),
        )
    )

    assert provider.requests[0].images[0].image.size == (64, 64)
    assert result.next_actions[0].data == {"x": 27, "y": 25}
    assert result.next_actions[0].target == "the red tile"


def test_agent_game_updater_history_renders_action6_in_model_space(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "probing_strategy": "choose action one",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    updater.update_agent_probing_context(
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
            context_history=_history_summary("world updated"),
            action_history=(
                ActionHistoryEntry(
                    action=ActionSpec(
                        action_id=GameAction.ACTION6,
                        data={"x": 32, "y": 43},
                        target="the lower middle tile",
                    ),
                    controllable=True,
                    changed_pixel_count=4,
                    change_summary="Targeted a point.",
                    action_mode="probing",
                ),
            ),
        )
    )

    text = provider.requests[0].text
    assert 'ACTION6 target="the lower middle tile"' in text
    assert '{"x": 500, "y": 696}' not in text
    assert "[changed_pixels=4%]" in text


def test_general_updater_updates_agent_general_context(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    provider = FakeUpdaterProvider(
        json.dumps({"updated_context": "new general context"})
    )
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="agent",
            previous_context=RoleContext(general="old K", game="live L"),
            run_id="run-1",
            game_id="game-1",
        )
    )

    request = provider.requests[0]
    assert request.target.role == "agent"
    assert request.target.task == "general"
    assert request.output_schema == updated_context_json_schema()
    assert result == RoleContext(general="new general context", game="live L")


def test_agent_game_updater_repairs_invalid_output(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    repaired = {
        "probing_strategy": "repair probing plan",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider("{}", json.dumps(repaired))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=1),
    )

    result = updater.update_agent_probing_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            context_history=_history_summary("world repaired"),
        )
    )

    assert len(provider.requests) == 2
    assert json.loads(result.context) == {
        "probing_strategy": "repair probing plan"
    }


def test_agent_probing_updater_falls_back_after_repair_exhaustion(
    tmp_path,
    caplog,
) -> None:
    _write_instruction_files(tmp_path)
    provider = FakeUpdaterProvider("{}")
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=0),
    )

    with caplog.at_level("ERROR"):
        result = updater.update_agent_probing_context(
            AgentGameContextUpdateInput(
                previous_context=RoleContext(
                    game=json.dumps(
                        {
                            "probing_strategy": "keep probing the center",
                            "policy_strategy": "reach target",
                        }
                    )
                ),
                current_observation=Observation(
                    id="obs-64",
                    step=1,
                    frame=Image.new("RGB", (64, 64), color=(1, 2, 3)),
                ),
                allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
                glossary_actions=(ActionSpec(action_id=GameAction.ACTION6),),
                context_history=_history_summary("world fallback"),
                actions_window=2,
            )
        )

    assert json.loads(result.context) == {
        "probing_strategy": "keep probing the center"
    }
    assert len(result.next_actions) == 2
    assert result.next_actions[0].name == "ACTION6"
    assert result.next_actions[0].data == {"x": 32, "y": 32}
    assert result.next_actions[0].target == ""
    assert "agent probing updater structured output repair exhausted" in caplog.text


def test_agent_policy_updater_fallback_returns_action_with_empty_previous(
    tmp_path,
    caplog,
) -> None:
    _write_instruction_files(tmp_path)
    provider = FakeUpdaterProvider("{}")
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=0),
    )

    with caplog.at_level("ERROR"):
        result = updater.update_agent_policy_context(
            AgentGameContextUpdateInput(
                previous_context=RoleContext(game=""),
                current_observation=_observation(),
                allowed_actions=(ActionSpec(action_id="ACTION1"),),
                glossary_actions=(ActionSpec(action_id="ACTION1"),),
                context_history=_history_summary("world fallback"),
            )
        )

    assert json.loads(result.context) == {"policy_strategy": ""}
    assert result.next_actions == (ActionSpec(action_id="ACTION1"),)
    assert result.updater_mode == "policy"
    assert "agent policy updater structured output repair exhausted" in caplog.text


def test_agent_game_updater_uses_only_allowed_actions_for_glossary(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "probing_strategy": "choose action one",
        "next_actions": [{"action_id": "ACTION1"}],
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    updater.update_agent_probing_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(
                ActionSpec(action_id="ACTION1"),
                ActionSpec(action_id="ACTION2"),
            ),
            context_history=_history_summary("world updated"),
        )
    )

    instructions = provider.requests[0].instructions
    assert "- `ACTION1`: up." in instructions
    assert "- `ACTION2`" not in instructions


def test_agent_game_parser_rejects_missing_field() -> None:
    with pytest.raises(UpdaterOutputError, match="missing keys"):
        parse_agent_game_updated_context_output(
            json.dumps(
                {
                    "next_actions": [{"action_id": "ACTION1"}],
                }
            ),
            mode="policy",
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
        )


def test_agent_game_schema_uses_next_actions_window() -> None:
    schema = agent_game_updated_context_json_schema(
        mode="probing",
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
        actions_window=3,
    )

    assert schema["required"] == ["probing_strategy", "next_actions"]
    next_actions_schema = schema["properties"]["next_actions"]
    assert next_actions_schema["minItems"] == 3
    assert next_actions_schema["maxItems"] == 3


def test_agent_game_schema_requires_action6_target_bbox_and_rgb() -> None:
    schema = agent_game_updated_context_json_schema(
        mode="probing",
        allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
    )

    action_schema = schema["properties"]["next_actions"]["items"]
    assert action_schema["required"] == [
        "action_id",
        "target",
        "bbox",
        "target_rgb_color",
    ]
    assert action_schema["properties"]["target"]["type"] == "string"
    assert action_schema["properties"]["bbox"]["minItems"] == 4
    assert action_schema["properties"]["target_rgb_color"]["maxItems"] == 3


def test_agent_probing_parser_accepts_summary_and_next_actions() -> None:
    context, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "probing_strategy": "probe action one",
                "next_actions": [{"action_id": "ACTION1"}],
            }
        ),
        mode="probing",
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert json.loads(context) == {"probing_strategy": "probe action one"}
    assert next_actions[0].name == "ACTION1"


def test_agent_probing_parser_accepts_multi_action_window() -> None:
    _, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "probing_strategy": "probe a short sequence",
                "next_actions": [
                    {"action_id": "ACTION1"},
                    {"action_id": "ACTION2"},
                ],
            }
        ),
        mode="probing",
        allowed_actions=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
        ),
        actions_window=2,
    )

    assert tuple(action.name for action in next_actions) == ("ACTION1", "ACTION2")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "probing_strategy": "old shape",
                "next_action": {"action_id": "ACTION1"},
            },
            "unexpected keys",
        ),
        (
            {"probing_strategy": "empty", "next_actions": []},
            "must not be empty",
        ),
        (
            {
                "probing_strategy": "too few",
                "next_actions": [
                    {"action_id": "ACTION1"},
                ],
            },
            "expected exactly the 2 action window",
        ),
        (
            {
                "probing_strategy": "too many",
                "next_actions": [
                    {"action_id": "ACTION1"},
                    {"action_id": "ACTION1"},
                ],
            },
            "expected exactly the 1 action window",
        ),
        (
            {
                "probing_strategy": "invalid action",
                "next_actions": [{"action_id": "ACTION2"}],
            },
            "invalid",
        ),
        (
            {
                "probing_strategy": "extra",
                "next_actions": [{"action_id": "ACTION1"}],
                "extra": "nope",
            },
            "unexpected keys",
        ),
        (
            {
                "probing_strategy": "simple target",
                "next_actions": [{"action_id": "ACTION1", "target": "a tile"}],
            },
            "invalid",
        ),
    ],
)
def test_agent_game_parser_rejects_invalid_next_actions(
    payload: dict[str, Any],
    match: str,
) -> None:
    actions_window = 2 if payload.get("probing_strategy") == "too few" else 1
    with pytest.raises(UpdaterOutputError, match=match):
        parse_agent_game_updated_context_output(
            json.dumps(payload),
            mode="probing",
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            actions_window=actions_window,
        )


def test_agent_game_parser_accepts_action6_bbox_and_rgb_targeting() -> None:
    _, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "probing_strategy": "probe center",
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
        mode="probing",
        allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
        arc_grid_crop_edges=(4, 4, 4, 4),
    )

    assert next_actions[0].data == {
        "bbox": [0, 800, 200, 1000],
        "target_rgb_color": [10, 20, 30],
    }
    assert next_actions[0].target == "the lower left tile"


def test_agent_game_parser_rejects_action6_without_target() -> None:
    with pytest.raises(UpdaterOutputError, match="invalid"):
        parse_agent_game_updated_context_output(
            json.dumps(
                {
                    "probing_strategy": "probe center",
                    "next_actions": [
                        {
                            "action_id": "ACTION6",
                            "bbox": [0, 800, 200, 1000],
                            "target_rgb_color": [10, 20, 30],
                        }
                    ],
                }
            ),
            mode="probing",
            allowed_actions=(ActionSpec(action_id=GameAction.ACTION6),),
        )


def test_agent_game_parser_accepts_top_level_policy_strategy() -> None:
    context, next_actions = parse_agent_game_updated_context_output(
        json.dumps(
            {
                "policy_strategy": "reach the exit",
                "next_actions": [{"action_id": "ACTION1"}],
            }
        ),
        mode="policy",
        allowed_actions=(ActionSpec(action_id="ACTION1"),),
    )

    assert json.loads(context) == {"policy_strategy": "reach the exit"}
    assert next_actions[0].name == "ACTION1"


def _history_summary(world_description: str) -> AgentContextHistorySummary:
    return AgentContextHistorySummary(
        world_description=world_description,
        action_effects={"ACTION1": "moves up"},
        updater_mode="probing",
        probing_evolution="probing evolved",
        policy_evolution="policy evolved",
    )


def _write_instruction_files(path) -> None:
    instruction_text = (
        "agent instructions\n"
        "- `Same past state detected`: if this is not empty, it means you were "
        "in this exact state before, meaning all you did was simply running in "
        "circle. This input intentionally contains only strategy fields."
    )
    (path / "agent_probing_context_updater_prompt.md").write_text(
        instruction_text,
        encoding="utf-8",
    )
    (path / "agent_policy_context_updater_prompt.md").write_text(
        instruction_text,
        encoding="utf-8",
    )
    (path / "agent_general_context_updater_prompt.md").write_text(
        "agent general instructions",
        encoding="utf-8",
    )


def _observation() -> Observation:
    return Observation(
        id="obs-1",
        step=1,
        frame=Image.new("RGB", (8, 8), color=(1, 2, 3)),
    )
