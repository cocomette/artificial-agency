"""Tests for debug playback with the active runtime shape."""

from __future__ import annotations

import json

from PIL import Image

from debug.playback.runtime import (
    PlaybackRequest,
    load_replay_rows,
    prepare_playback,
)
from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    Observation,
    ObservationRef,
    RoleContext,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models import ModelRegistry, UpdaterTaskRegistry
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
)
from face_of_agi.models.compacter import AgentCompacterSummary


class LiveAgent:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> DecisionResult:
        self.calls += 1
        action = ActionSpec(action_id="ACTION2")
        observation = kwargs.get("current_observation") or args[1]
        ref = ObservationRef(memory="state", id=observation.id)
        return DecisionResult(
            final_action=action,
            trace=AgentTrace(
                step=observation.step,
                first_observation_ref=ref,
                current_observation_ref=ref,
                final_action=action,
            ),
        )


class LiveChangeSummary:
    def summarize(self, *args, **kwargs) -> ChangeSummaryResult:
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="element",
                    element_description="visible object",
                    element_mutation="changed",
                ),
            ),
            change_detected=True,
            metadata={},
        )


class LiveCompacter:
    def compact_agent_context(self, *args, **kwargs) -> AgentCompacterSummary:
        return AgentCompacterSummary(
            world_description="live world",
            action_effects={},
            previous_actions_summary="live actions",
            previous_strategy_summary="live strategies",
        )


class LiveAgentUpdater:
    def __init__(self) -> None:
        self.calls = 0

    def update_agent_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.calls += 1
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {
                    "current_strategy": f"live-agent-{self.calls}",
                }
            ),
            next_actions=(update_input.allowed_actions[0],),
        )


def test_load_replay_rows_returns_prior_complete_rows(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    _write_state(memory, turn_id=1, action_name="ACTION1", agent_game="source-1")
    _write_state(memory, turn_id=2, action_name="ACTION2", agent_game="source-2")

    rows = load_replay_rows(
        memory,
        PlaybackRequest(source_run_id="source-run", game_id="game-1", turn_id=2),
    )

    assert [row.metadata["turn_id"] for row in rows] == [1]
    assert json.loads(rows[0].agent_context.game)["current_strategy"] == (
        _mechanics("source-1")
    )


def test_prepare_playback_replays_agent_decision_and_agent_context(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    _write_state(memory, turn_id=1, action_name="ACTION1", agent_game="source-1")
    _write_state(memory, turn_id=2, action_name="ACTION2", agent_game="source-2")
    live_agent = LiveAgent()
    live_agent_updater = LiveAgentUpdater()

    setup = prepare_playback(
        state_memory=memory,
        request=PlaybackRequest(
            source_run_id="source-run",
            game_id="game-1",
            turn_id=2,
        ),
        live_models=ModelRegistry(
            orchestrator_agent=live_agent,
            change_summary_model=LiveChangeSummary(),
            compacter=LiveCompacter(),
            updater_tasks=UpdaterTaskRegistry(
                agent_updater=live_agent_updater,
            ),
        ),
    )

    agent = setup.models.require_orchestrator_agent()
    decision = agent.decide(
        context=RoleContext(game="live"),
        current_observation=_observation("obs-live"),
        action_space=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
        ),
        glossary_actions=(
            ActionSpec(action_id="ACTION1"),
            ActionSpec(action_id="ACTION2"),
        ),
    )
    updater = setup.models.require_updater_tasks().require_agent_updater()
    replay_result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation("obs-live"),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
        )
    )
    live_result = updater.update_agent_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation("obs-live"),
            allowed_actions=(ActionSpec(action_id="ACTION2"),),
            glossary_actions=(ActionSpec(action_id="ACTION2"),),
        )
    )

    assert decision.final_action.name == "ACTION1"
    assert live_agent.calls == 0
    assert json.loads(replay_result.context) == {
        "current_strategy": _mechanics("source-1"),
    }
    assert replay_result.next_actions[0].name == "ACTION1"
    assert json.loads(live_result.context) == {
        "current_strategy": "live-agent-1",
    }
    assert live_agent_updater.calls == 1


def _write_state(
    memory: StateMemory,
    *,
    turn_id: int,
    action_name: str,
    agent_game: str,
) -> None:
    observation = _observation(f"obs-{turn_id}")
    action = ActionSpec(action_id=action_name)
    memory.write_state(
        run_id="source-run",
        game_id="game-1",
        step=turn_id,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(
            agent=RoleContext(game=_agent_game_context(agent_game))
        ),
        agent_trace=_trace(observation, action),
        metadata={
            "turn_id": turn_id,
            "control_mode": {
                "controllable": True,
                "reason": "real_environment_turn",
            },
        },
    )


def _agent_game_context(label: str) -> str:
    return json.dumps(
        {
            "current_strategy": _mechanics(label),
        },
        indent=2,
    )


def _mechanics(label: str) -> str:
    return label


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(1, 2, 3)),
    )


def _trace(observation: Observation, action: ActionSpec) -> AgentTrace:
    ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=action,
    )
