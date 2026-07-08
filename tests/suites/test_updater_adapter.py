"""Tests for the active updater P prompt adapter tasks."""

from __future__ import annotations

import json
from typing import Any

import pytest
from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    Observation,
    RoleContext,
)
from face_of_agi.models.updater import (
    AGENT_GAME_CONTEXT_KEYS,
    AGENT_GAME_CONTEXT_MAX_CHARS,
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
        task="agent_game",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_game_context_updater_prompt.md"
    assert updater_instruction_path(
        task="general",
        role="agent",
        instruction_dir=tmp_path,
    ) == tmp_path / "agent_general_context_updater_prompt.md"


def test_agent_game_updater_returns_structured_agent_context(tmp_path) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "updated_context": {
            key: f"{key} updated" for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), frame_scale=1),
    )

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=3,
        )
    )

    request = provider.requests[0]
    assert request.target.role == "agent"
    assert request.target.task == "agent_game"
    assert request.output_schema == agent_game_updated_context_json_schema()
    assert "## Action glossary" in request.instructions
    assert request.images[0].label == "current_observation_frame"
    assert json.loads(result.game) == response["updated_context"]


def test_agent_game_updater_prompt_renders_action_history_relative_to_crop(
    tmp_path,
) -> None:
    _write_instruction_files(tmp_path)
    response = {
        "updated_context": {
            key: f"{key} updated" for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    provider = FakeUpdaterProvider(json.dumps(response))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(
            instruction_dir=str(tmp_path),
            frame_scale=1,
            input_image_crop_arc_grid_edges=4,
        ),
    )

    updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K", game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION6"),),
            glossary_actions=(ActionSpec(action_id="ACTION6"),),
            action_history_window=1,
            action_history=(
                ActionHistoryEntry(
                    action=ActionSpec(action_id="ACTION6", data={"x": 4, "y": 60}),
                    controllable=True,
                    changed_pixel_percent=0,
                    change_summary="no changes",
                ),
            ),
        )
    )

    assert (
        '1. ACTION6 {"x": 0, "y": 1000} [latest] [changed_pixel_percent=0] '
        "change: no changes"
    ) in provider.requests[0].text


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
        "updated_context": {
            key: f"{key} repaired" for key in AGENT_GAME_CONTEXT_KEYS
        }
    }
    provider = FakeUpdaterProvider("{}", json.dumps(repaired))
    updater = PromptUpdaterAdapter(
        provider,
        UpdaterConfig(instruction_dir=str(tmp_path), repair_attempts=1),
    )

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation(),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=1,
        )
    )

    assert len(provider.requests) == 2
    assert json.loads(result.game) == repaired["updated_context"]


def test_agent_game_parser_rejects_missing_field() -> None:
    with pytest.raises(UpdaterOutputError, match="missing keys"):
        parse_agent_game_updated_context_output(
            json.dumps({"updated_context": {"goals": "only goals"}})
        )


def test_agent_game_parser_rejects_context_over_character_cap() -> None:
    oversized = {key: "" for key in AGENT_GAME_CONTEXT_KEYS}
    oversized["history"] = "x" * AGENT_GAME_CONTEXT_MAX_CHARS

    with pytest.raises(UpdaterOutputError, match="12000 character cap"):
        parse_agent_game_updated_context_output(
            json.dumps({"updated_context": oversized})
        )


def _write_instruction_files(path) -> None:
    (path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
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
