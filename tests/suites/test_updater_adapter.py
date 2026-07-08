"""Tests for current updater model P adapters."""

from __future__ import annotations

import json

from PIL import Image
import pytest

from face_of_agi.contracts import ActionSpec, Observation, RoleContext
from face_of_agi.models.historizer import AgentContextHistorySummary
from face_of_agi.models.memory import GameMemoryDocument
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentProgressFeedback,
    GeneralKnowledgeUpdateInput,
    PromptUpdateProviderResponse,
    PromptUpdateRequest,
    PromptUpdaterAdapter,
    UpdaterConfig,
    UpdaterOutputError,
    agent_game_updated_context_json_schema,
    parse_agent_game_updated_context_output,
    parse_updated_context_output,
    updated_context_json_schema,
)


class FakePromptUpdaterProvider:
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
        return PromptUpdateProviderResponse(
            target=request.target,
            text=self.responses.pop(0),
            metadata={"response": "ok"},
        )

    def repair_prompt(
        self,
        request: PromptUpdateRequest,
        *,
        invalid_text: str,
        validation_error: str,
        attempt: int,
    ) -> PromptUpdateProviderResponse:
        del invalid_text, validation_error, attempt
        self.requests.append(request)
        return PromptUpdateProviderResponse(
            target=request.target,
            text=self.responses.pop(0),
            metadata={"response": "repair"},
        )


def test_prompt_updater_updates_agent_game_context(tmp_path) -> None:
    (tmp_path / "agent_game_context_updater_prompt.md").write_text(
        "agent game instructions",
        encoding="utf-8",
    )
    provider = FakePromptUpdaterProvider(
        json.dumps(
            {
                "updated_context": {
                    "goals": "reach the door",
                    "game_mechanics": "ACTION1 moves up",
                    "policy": "try ACTION1 next",
                    "history": "ACTION2 did not help",
                    "extras": "none",
                }
            }
        )
    )
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(general="K^X", game="L^X"),
            current_observation=Observation(
                id="obs-1",
                step=1,
                frame=Image.new("RGB", (4, 4), color=(0, 0, 0)),
            ),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=5,
            game_memory=GameMemoryDocument("ACTION1 opened a path."),
            context_history=AgentContextHistorySummary.not_available(),
            turn_metrics=AgentProgressFeedback(cumulative_score=1.0),
        )
    )

    assert result.general == "K^X"
    assert "reach the door" in result.game
    assert provider.requests[0].target.task == "agent_game"
    assert provider.requests[0].metadata["agent_game_context_max_chars"] == 12000
    assert "ACTION1 opened a path." in provider.requests[0].text


def test_prompt_updater_updates_general_context(tmp_path) -> None:
    (tmp_path / "agent_general_context_updater_prompt.md").write_text(
        "agent general instructions",
        encoding="utf-8",
    )
    provider = FakePromptUpdaterProvider(
        json.dumps({"updated_context": "updated K^X"})
    )
    updater = PromptUpdaterAdapter(
        provider=provider,
        config=UpdaterConfig(instruction_dir=str(tmp_path)),
    )

    result = updater.update_general_knowledge(
        GeneralKnowledgeUpdateInput(
            role="agent",
            previous_context=RoleContext(general="K^X", game="L^X"),
            run_id="run-1",
            game_id="game-1",
            stop_reason="finished",
        )
    )

    assert result == RoleContext(general="updated K^X", game="L^X")
    assert provider.requests[0].target.task == "general"
    assert provider.requests[0].metadata["general_context_max_chars"] == 20000


def test_updater_output_schemas_include_configurable_caps() -> None:
    assert updated_context_json_schema(
        general_context_max_chars=12
    )["properties"]["updated_context"]["maxLength"] == 12
    field_schema = agent_game_updated_context_json_schema(
        agent_game_context_field_max_chars=8
    )["properties"]["updated_context"]["properties"]["goals"]
    assert field_schema["maxLength"] == 8


def test_updater_parsers_reject_oversized_outputs() -> None:
    with pytest.raises(UpdaterOutputError, match="character cap"):
        parse_updated_context_output(
            json.dumps({"updated_context": "x" * 6}),
            max_chars=5,
        )

    payload = {
        "updated_context": {
            "goals": "x" * 6,
            "game_mechanics": "ok",
            "policy": "ok",
            "history": "ok",
            "extras": "ok",
        }
    }
    with pytest.raises(UpdaterOutputError, match="fields exceed"):
        parse_agent_game_updated_context_output(
            json.dumps(payload),
            field_max_chars=5,
        )
