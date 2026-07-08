"""Tests for debug playback with the active runtime shape."""

from __future__ import annotations

from PIL import Image

from debug.playback.runtime import (
    PlaybackRequest,
    load_replay_rows,
    prepare_playback,
)
from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
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
    GeneralKnowledgeUpdateInput,
)


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
            summary="changed",
            changed_pixel_percent=1,
            change_detected=True,
            metadata={},
        )


class LiveAgentUpdater:
    def __init__(self) -> None:
        self.calls = 0

    def update_agent_game_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> RoleContext:
        self.calls += 1
        return RoleContext(game=f"live-agent-{self.calls}")


class LiveGeneralUpdater:
    def update_general_knowledge(
        self,
        update_input: GeneralKnowledgeUpdateInput,
    ) -> RoleContext:
        return RoleContext(general="live-general", game=update_input.previous_context.game)


def test_load_replay_rows_returns_prior_complete_rows(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    _write_state(memory, turn_id=1, action_name="ACTION1", agent_game="source-1")
    _write_state(memory, turn_id=2, action_name="ACTION2", agent_game="source-2")

    rows = load_replay_rows(
        memory,
        PlaybackRequest(source_run_id="source-run", game_id="game-1", turn_id=2),
    )

    assert [row.metadata["turn_id"] for row in rows] == [1]
    assert rows[0].agent_context.game == "source-1"


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
            updater_tasks=UpdaterTaskRegistry(
                agent_game_updater=live_agent_updater,
                general_updater=LiveGeneralUpdater(),
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
    updater = setup.models.require_updater_tasks().require_agent_game_updater()
    replay_context = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation("obs-live"),
            allowed_actions=(ActionSpec(action_id="ACTION1"),),
            glossary_actions=(ActionSpec(action_id="ACTION1"),),
            action_history_window=1,
        )
    )
    live_context = updater.update_agent_game_context(
        AgentGameContextUpdateInput(
            previous_context=RoleContext(game="old"),
            current_observation=_observation("obs-live"),
            allowed_actions=(ActionSpec(action_id="ACTION2"),),
            glossary_actions=(ActionSpec(action_id="ACTION2"),),
            action_history_window=1,
        )
    )

    assert decision.final_action.name == "ACTION1"
    assert live_agent.calls == 0
    assert replay_context.game == "source-1"
    assert live_context.game == "live-agent-1"
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
        contexts=ContextDocuments(agent=RoleContext(game=agent_game)),
        agent_trace=_trace(observation, action),
        metadata={
            "turn_id": turn_id,
            "control_mode": {
                "controllable": True,
                "reason": "real_environment_turn",
            },
        },
    )


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
