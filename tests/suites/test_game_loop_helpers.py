"""Tests for small game-loop helper contracts."""

from dataclasses import asdict
import json
from types import SimpleNamespace

from arcengine import GameState
from PIL import Image
import pytest

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryResetMarker,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    EnvironmentInfo,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
    RoleContext,
    RuntimeConfig,
    UpdaterFrameTransitionInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.change import ChangeSummaryResult
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    UpdaterTaskRegistry,
)
from face_of_agi.models.compacter import AgentCompacterInput, AgentCompacterSummary
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_agent_context_update,
)
from face_of_agi.orchestration.game_loop.actions.steps import (
    LEVEL_SOLVED_RESET_NOTICE,
    attach_change_summary,
    bootstrap_agent_updater_decision,
    has_observed_transition,
    prepare_observed_transition,
    resolve_next_snapshot,
    run_compacter,
    run_updaters,
    summarize_change,
    summarize_change_model,
)
from face_of_agi.orchestration.game_loop.helpers import build_action_history_entry
from face_of_agi.orchestration.game_loop.lifecycle import (
    check_lifecycle,
    reset_after_game_over,
)
from face_of_agi.orchestration.game_loop.persistence import persist_turn_shell
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)
from face_of_agi.orchestration.game_loop.simulation import (
    KnownStateTransitionEdge,
    SIMULATED_ROW_KEY,
    SIMULATION_CATCHUP_KEY,
    SimulationCatchupPlan,
    _edge_for_action,
    _finish_simulation,
    _known_state_transition_edges,
    _next_simulated_action,
    _simulation_catchup_plan,
    _simulation_action_history_entries,
    maybe_run_known_state_simulation,
)
from face_of_agi.frames import observation_frame_hash


def test_action_history_changed_pixel_count_uses_change_summary_crop() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))

    entry = build_action_history_entry(
        frame_context=_frame_context(previous_frame),
        final_action=ActionSpec(action_id="ACTION1"),
        next_observation=Observation(id="current", step=1, frame=current_frame),
        change_summary="Only the cropped-away border changed.",
        change_summary_crop_edges=4,
    )

    assert entry.changed_pixel_count == 0


def test_persist_turn_stores_completed_action_history(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    action_2 = ActionSpec(action_id="ACTION2")
    action_6 = ActionSpec(
        action_id="ACTION6",
        data={"x": 55, "y": 47},
        target="yellow L-shape",
        target_value=11,
    )
    observation = _colored_observation("current", (1, 2, 3))
    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action_2, action_6)),
        contexts=contexts,
    )
    ref = ObservationRef(memory="state", id=observation.id)
    prior_entry = ActionHistoryEntry(
        action=action_2,
        controllable=True,
        changed_pixel_count=3.2105,
        change_summary="old movement",
        action_count=12,
    )
    current_entry = ActionHistoryEntry(
        action=action_6,
        controllable=True,
        changed_pixel_count=0.0,
        change_summary="No changes happened for this transition.",
        action_count=13,
    )
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        current_source_state_id=source.id,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action_2, action_6)),
        recent_action_history=(prior_entry,),
    )

    persist_turn_shell(
        frame_context=frame_context,
        turn_id=1,
        decision=SimpleNamespace(final_action=action_6, trace=_trace(observation)),
        update_input=UpdaterFrameTransitionInput(
            current_observation_ref=ref,
            actual_next_observation_ref=ref,
            decision_trace=_trace(observation),
            actual_next_observation=observation,
            action_history_entry=current_entry,
        ),
        state_record_ids=[],
        state_memory=memory,
        contexts=contexts,
        debug=DebugBus.disabled(),
    )

    completed = memory.list_states(game_id="game-1")[0]
    actions = [
        item["action"]["action_id"]
        for item in completed.metadata["recent_action_history"]
    ]
    assert actions == ["ACTION2", "ACTION6"]


def test_persist_turn_allows_no_current_action_history_entry(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    action = ActionSpec(action_id="ACTION1")
    observation = _colored_observation("current", (1, 2, 3))
    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=contexts,
    )
    ref = ObservationRef(memory="state", id=observation.id)
    frame_context = FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        current_source_state_id=source.id,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        recent_action_history=(),
    )

    persist_turn_shell(
        frame_context=frame_context,
        turn_id=1,
        decision=SimpleNamespace(final_action=action, trace=_trace(observation)),
        update_input=UpdaterFrameTransitionInput(
            current_observation_ref=ref,
            actual_next_observation_ref=ref,
            decision_trace=_trace(observation),
            actual_next_observation=observation,
        ),
        state_record_ids=[],
        state_memory=memory,
        contexts=contexts,
        debug=DebugBus.disabled(),
    )

    completed = memory.list_states(game_id="game-1")[0]
    assert completed.metadata["recent_action_history"] == []


def test_known_state_simulation_replays_matching_single_action(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    contexts = ContextDocuments(
        agent=RoleContext(
            game=json.dumps(
                {
                    "current_strategy": "",
                }
            )
        )
    )
    control_mode = FrameControlMode.real_environment_turn((action_1, action_2))
    first = _colored_observation("known-a", (1, 2, 3))
    successor = _colored_observation("known-b", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    first_hash = observation_frame_hash(first, crop_edges=crop_edges)
    successor_hash = observation_frame_hash(successor, crop_edges=crop_edges)
    history_entry = ActionHistoryEntry(
        action=action_1,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="historical action moved to B",
        action_count=99,
    )
    pending_history_entry = ActionHistoryEntry(
        action=action_2,
        controllable=True,
        changed_pixel_count=2.0,
        change_summary="current transition before simulation",
        action_count=3,
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_1,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": first_hash,
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=successor,
        chosen_action=action_2,
        contexts=contexts,
        agent_trace=_trace(successor),
        metadata={
            "current_frame_hash": successor_hash,
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [asdict(history_entry)],
        },
    )
    current_source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        current_observation=first,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        contexts=contexts,
        current_frame_hash=first_hash,
        current_frame_hash_crop_edges=crop_edges,
    )
    environment = _FakeEnvironment((successor,))
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=first,
        remaining_actions=10,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
        queued_updater_actions=(action_1,),
        real_step_count=3,
    )
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=ObservationRef(memory="state", id=first.id),
        actual_next_observation_ref=ObservationRef(memory="state", id=first.id),
        decision_trace=_trace(first),
        actual_next_observation=first,
        action_history_entry=pending_history_entry,
    )
    session.compacter_context_summary = AgentCompacterSummary(
        world_description="simulated world",
        special_events="none",
        action_effects={"ACTION1": "historical move"},
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        observation=first,
        observation_ref=ObservationRef(memory="state", id=first.id),
        source_state_id=current_source.id,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )
    updater = _SequencedAgentUpdater((action_2,))
    compacter = _RecordingCompacter(
        previous_actions_summary="simulation route",
        previous_strategy_summary="simulation strategy",
    )

    consumed = maybe_run_known_state_simulation(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        compacter=compacter,
        state_memory=memory,
        debug=DebugBus(state_memory=memory),
    )

    assert consumed is True
    assert [action.name for action in session.queued_updater_actions] == [
        "ACTION2",
    ]
    assert [action.name for action in environment.submitted_actions] == ["ACTION1"]
    assert session.remaining_actions == 9
    assert session.real_step_count == 4
    assert session.latest_environment_observation == successor
    assert len(session.frame_buffer) == 1
    assert session.frame_buffer[0].id == successor.id
    assert (
        observation_frame_hash(session.frame_buffer[0], crop_edges=crop_edges)
        == successor_hash
    )
    assert session.previous_observation is None
    assert session.last_decision is None
    assert session.current is None
    assert [entry.change_summary for entry in session.action_history] == [
        "current transition before simulation",
        "historical action moved to B",
    ]
    assert [entry.action_count for entry in session.action_history] == [3, 4]
    assert compacter.inputs
    assert updater.inputs[0].action_history == tuple(session.action_history)
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["successful"] is True
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["catchup_actions"] == (
        "ACTION1",
    )
    simulated_rows = [
        row
        for row in memory.list_states(game_id="game-1")
        if row.metadata.get(SIMULATED_ROW_KEY)
    ]
    assert len(simulated_rows) == 2
    first_simulated_history = simulated_rows[0].metadata["recent_action_history"]
    assert [item["action"]["action_id"] for item in first_simulated_history] == [
        "ACTION2",
        "ACTION1",
    ]
    assert [item["action_count"] for item in first_simulated_history] == [3, 4]
    assert (
        first_simulated_history[-1]["change_summary"]
        == "historical action moved to B"
    )
    exit_row_history = simulated_rows[1].metadata["recent_action_history"]
    assert [item["action"]["action_id"] for item in exit_row_history] == [
        "ACTION2",
        "ACTION1",
    ]
    assert SIMULATION_CATCHUP_KEY not in simulated_rows[0].metadata
    assert simulated_rows[1].metadata[SIMULATION_CATCHUP_KEY]["successful"] is True
    assert simulated_rows[1].metadata[SIMULATION_CATCHUP_KEY]["catchup_actions"] == [
        "ACTION1",
    ]
    assert simulated_rows[1].metadata[SIMULATION_CATCHUP_KEY]["exit_action"] == (
        "ACTION2"
    )
    debug_records = memory.list_model_input_debug_records(game_id="game-1")
    assert [record.call_slot for record in debug_records] == ["compacter"]


def test_known_state_edges_use_successor_latest_history_snapshot(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    first = _colored_observation("known-a", (1, 2, 3))
    successor = _colored_observation("known-b", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    old_nonmatching_entry = ActionHistoryEntry(
        action=action_2,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="old nonmatching action entry",
    )
    latest_successor_entry = ActionHistoryEntry(
        action=action_1,
        controllable=True,
        changed_pixel_count=2.0,
        change_summary="latest successor row entry",
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_1,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": observation_frame_hash(
                first,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=successor,
        chosen_action=action_2,
        contexts=contexts,
        agent_trace=_trace(successor),
        metadata={
            "current_frame_hash": observation_frame_hash(
                successor,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [
                asdict(old_nonmatching_entry),
                asdict(latest_successor_entry),
            ],
        },
    )

    edges = _known_state_transition_edges(
        memory,
        game_id="game-1",
        run_id="run-1",
        before_state_id=999,
    )

    assert len(edges) == 1
    assert edges[0].action == action_1
    assert tuple(entry.change_summary for entry in edges[0].action_history_entries) == (
        "latest successor row entry",
    )


def test_known_state_edges_reject_mismatched_successor_history(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_6 = ActionSpec(
        action_id="ACTION6",
        data={"x": 55, "y": 47},
        target="yellow L-shape",
        target_value=11,
    )
    action_2 = ActionSpec(action_id="ACTION2")
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    first = _colored_observation("known-a", (1, 2, 3))
    successor = _colored_observation("known-b", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_6,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": observation_frame_hash(
                first,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=successor,
        chosen_action=action_6,
        contexts=contexts,
        agent_trace=_trace(successor),
        metadata={
            "current_frame_hash": observation_frame_hash(
                successor,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [
                asdict(
                    ActionHistoryEntry(
                        action=action_2,
                        controllable=True,
                        changed_pixel_count=3.2105,
                        change_summary="wrong stored movement",
                    )
                ),
            ],
        },
    )

    with pytest.raises(RuntimeError, match="inconsistent action history"):
        _known_state_transition_edges(
            memory,
            game_id="game-1",
            run_id="run-1",
            before_state_id=999,
        )


def test_known_state_edges_skip_game_over_reset_boundary(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_4 = ActionSpec(action_id="ACTION4")
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    before_game_over = _colored_observation("before-game-over", (1, 2, 3))
    game_over = _colored_observation("game-over", (4, 5, 6))
    after_reset = _colored_observation("after-reset", (7, 8, 9))
    crop_edges = (0, 0, 0, 0)
    latest_action_entry = ActionHistoryEntry(
        action=action_4,
        controllable=True,
        changed_pixel_count=3.2105,
        change_summary="last real action before game over",
    )
    reset_marker = {
        "type": "game_reset",
        **asdict(ActionHistoryResetMarker(reason="game_over_reset", restart_count=1)),
    }
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=before_game_over,
        chosen_action=action_4,
        contexts=contexts,
        agent_trace=_trace(before_game_over),
        metadata={
            "current_frame_hash": observation_frame_hash(
                before_game_over,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=game_over,
        chosen_action=ActionSpec.none(),
        contexts=contexts,
        agent_trace=_trace(game_over),
        metadata={
            "current_frame_hash": observation_frame_hash(
                game_over,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [asdict(latest_action_entry)],
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=after_reset,
        chosen_action=action_4,
        contexts=contexts,
        agent_trace=_trace(after_reset),
        metadata={
            "current_frame_hash": observation_frame_hash(
                after_reset,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [asdict(latest_action_entry), reset_marker],
        },
    )

    edges = _known_state_transition_edges(
        memory,
        game_id="game-1",
        run_id="run-1",
        before_state_id=999,
    )

    assert edges == ()


def test_known_state_edges_accept_matching_action6_successor_history(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_6 = ActionSpec(
        action_id="ACTION6",
        data={"x": 55, "y": 47},
        target="yellow L-shape",
        target_value=11,
    )
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    first = _colored_observation("known-a", (1, 2, 3))
    successor = _colored_observation("known-b", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_6,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": observation_frame_hash(
                first,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=successor,
        chosen_action=ActionSpec(action_id="ACTION2"),
        contexts=contexts,
        agent_trace=_trace(successor),
        metadata={
            "current_frame_hash": observation_frame_hash(
                successor,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [
                asdict(
                    ActionHistoryEntry(
                        action=action_6,
                        controllable=True,
                        changed_pixel_count=3.2105,
                        change_summary="matching stored ACTION6 movement",
                    )
                ),
            ],
        },
    )

    edges = _known_state_transition_edges(
        memory,
        game_id="game-1",
        run_id="run-1",
        before_state_id=999,
    )

    assert len(edges) == 1
    assert edges[0].action == action_6
    assert edges[0].action_history_entries[0].action == action_6


def test_known_state_edges_allow_empty_successor_history(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    first = _colored_observation("known-a", (1, 2, 3))
    successor = _colored_observation("known-b", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_1,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": observation_frame_hash(
                first,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=successor,
        chosen_action=action_2,
        contexts=contexts,
        agent_trace=_trace(successor),
        metadata={
            "current_frame_hash": observation_frame_hash(
                successor,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )

    edges = _known_state_transition_edges(
        memory,
        game_id="game-1",
        run_id="run-1",
        before_state_id=999,
    )

    assert len(edges) == 1
    assert edges[0].action == action_1
    assert edges[0].action_history_entries == ()


def test_known_state_edges_do_not_jump_over_simulated_rows(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action_6 = ActionSpec(
        action_id="ACTION6",
        data={"x": 55, "y": 47},
        target="yellow L-shape",
        target_value=11,
    )
    action_2 = ActionSpec(action_id="ACTION2")
    contexts = ContextDocuments(agent=RoleContext(game="{}"))
    first = _colored_observation("known-a", (1, 2, 3))
    simulated = _colored_observation("simulated", (2, 3, 4))
    later_real = _colored_observation("later-real", (4, 5, 6))
    crop_edges = (0, 0, 0, 0)
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=first,
        chosen_action=action_6,
        contexts=contexts,
        agent_trace=_trace(first),
        metadata={
            "current_frame_hash": observation_frame_hash(
                first,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=simulated,
        chosen_action=action_6,
        contexts=contexts,
        agent_trace=_trace(simulated),
        metadata={
            SIMULATED_ROW_KEY: True,
            "current_frame_hash": observation_frame_hash(
                simulated,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [
                asdict(
                    ActionHistoryEntry(
                        action=action_6,
                        controllable=True,
                        changed_pixel_count=3.2105,
                        change_summary="simulated movement",
                    )
                ),
            ],
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=3,
        frame_index=0,
        frame_count=1,
        current_observation=later_real,
        chosen_action=action_2,
        contexts=contexts,
        agent_trace=_trace(later_real),
        metadata={
            "current_frame_hash": observation_frame_hash(
                later_real,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": list(crop_edges),
            "recent_action_history": [
                asdict(
                    ActionHistoryEntry(
                        action=action_2,
                        controllable=True,
                        changed_pixel_count=1.0,
                        change_summary="later real movement",
                    )
                ),
            ],
        },
    )

    edges = _known_state_transition_edges(
        memory,
        game_id="game-1",
        run_id="run-1",
        before_state_id=999,
    )

    assert edges == ()


def test_finish_simulation_marks_catchup_mismatch_without_blocking_exit() -> None:
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    first = _colored_observation("known-a", (1, 2, 3))
    expected = _colored_observation("known-b", (4, 5, 6))
    actual = _colored_observation("known-c", (7, 8, 9))
    crop_edges = (0, 0, 0, 0)
    expected_hash = observation_frame_hash(expected, crop_edges=crop_edges)
    actual_hash = observation_frame_hash(actual, crop_edges=crop_edges)
    control_mode = FrameControlMode.real_environment_turn((action_1, action_2))
    environment = _FakeEnvironment((actual,))
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=first,
        remaining_actions=10,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        observation=first,
        observation_ref=ObservationRef(memory="state", id=first.id),
        source_state_id=1,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    _finish_simulation(
        session,
        debug=DebugBus.disabled(),
        turn_id=3,
        simulated_actions=(action_1,),
        catchup_plan=SimulationCatchupPlan(
            actions=(action_1,),
            source="direct_simulated_path",
        ),
        exit_action=action_2,
        expected_frame_hash=expected_hash,
        crop_edges=crop_edges,
        exit_reason="unknown_action",
        duration_seconds=1.0,
    )

    assert [action.name for action in environment.submitted_actions] == ["ACTION1"]
    assert [action.name for action in session.queued_updater_actions] == ["ACTION2"]
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["successful"] is False
    assert (
        session.turn_metadata[SIMULATION_CATCHUP_KEY]["actual_frame_hash"]
        == actual_hash
    )
    assert session.current is None
    assert session.process_turn is False


def test_finish_simulation_aborts_catchup_from_terminal_observation() -> None:
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    terminal = Observation(
        id="game-over",
        step=30,
        frame=None,
        raw_frame_data=SimpleNamespace(
            state=GameState.GAME_OVER,
            levels_completed=0,
        ),
    )
    control_mode = FrameControlMode.real_environment_turn((action_1, action_2))
    environment = _FakeEnvironment((_colored_observation("unused", (7, 8, 9)),))
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=terminal,
        remaining_actions=10,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        observation=terminal,
        observation_ref=ObservationRef(memory="state", id=terminal.id),
        source_state_id=1,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=ObservationRef(memory="state", id=terminal.id),
    )

    _finish_simulation(
        session,
        debug=DebugBus.disabled(),
        turn_id=3,
        simulated_actions=(action_1,),
        catchup_plan=SimulationCatchupPlan(
            actions=(action_1,),
            source="direct_simulated_path",
        ),
        exit_action=action_2,
        expected_frame_hash="expected",
        crop_edges=(0, 0, 0, 0),
        exit_reason="unknown_action",
        duration_seconds=1.0,
    )

    assert environment.submitted_actions == []
    assert session.queued_updater_actions == ()
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["successful"] is False
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["actual_frame_hash"] is None
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["aborted"] is True
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["abort_reason"] == "game_over"
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["catchup_actions"] == ()
    assert session.current is None
    assert session.process_turn is False


def test_finish_simulation_aborts_catchup_when_step_completes_level() -> None:
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    first = _colored_observation("known-a", (1, 2, 3))
    completed = Observation(
        id="completed-level",
        step=1,
        frame=None,
        raw_frame_data=SimpleNamespace(
            state=GameState.NOT_FINISHED,
            levels_completed=1,
        ),
    )
    control_mode = FrameControlMode.real_environment_turn((action_1, action_2))
    environment = _FakeEnvironment((completed,))
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=environment,
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=first,
        remaining_actions=10,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        observation=first,
        observation_ref=ObservationRef(memory="state", id=first.id),
        source_state_id=1,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=ObservationRef(memory="state", id=first.id),
    )

    _finish_simulation(
        session,
        debug=DebugBus.disabled(),
        turn_id=3,
        simulated_actions=(action_1,),
        catchup_plan=SimulationCatchupPlan(
            actions=(action_1,),
            source="direct_simulated_path",
        ),
        exit_action=action_2,
        expected_frame_hash="expected",
        crop_edges=(0, 0, 0, 0),
        exit_reason="unknown_action",
        duration_seconds=1.0,
    )

    assert [action.name for action in environment.submitted_actions] == ["ACTION1"]
    assert session.latest_environment_observation is completed
    assert session.queued_updater_actions == ()
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["successful"] is False
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["actual_frame_hash"] is None
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["aborted"] is True
    assert (
        session.turn_metadata[SIMULATION_CATCHUP_KEY]["abort_reason"]
        == "level_completed"
    )
    assert session.turn_metadata[SIMULATION_CATCHUP_KEY]["catchup_actions"] == (
        "ACTION1",
    )
    assert session.remaining_actions == 9


def test_simulation_catchup_plan_finds_shorter_historical_path() -> None:
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")
    action_3 = ActionSpec(action_id="ACTION3")
    action_4 = ActionSpec(action_id="ACTION4")
    action_5 = ActionSpec(action_id="ACTION5")
    edges = (
        _edge(
            source_state_id=1,
            successor_state_id=2,
            source_hash="entry",
            successor_hash="sim-a",
            action=action_1,
        ),
        _edge(
            source_state_id=2,
            successor_state_id=3,
            source_hash="sim-a",
            successor_hash="sim-b",
            action=action_2,
        ),
        _edge(
            source_state_id=3,
            successor_state_id=4,
            source_hash="sim-b",
            successor_hash="end",
            action=action_3,
        ),
        _edge(
            source_state_id=4,
            successor_state_id=5,
            source_hash="entry",
            successor_hash="shortcut",
            action=action_4,
        ),
        _edge(
            source_state_id=5,
            successor_state_id=6,
            source_hash="shortcut",
            successor_hash="end",
            action=action_5,
        ),
    )

    plan = _simulation_catchup_plan(
        edges=edges,
        entry_frame_hash="entry",
        simulated_end_frame_hash="end",
        simulated_actions=(action_1, action_2, action_3),
    )

    assert [action.name for action in plan.actions] == ["ACTION4", "ACTION5"]
    assert plan.source == "historical_graph"
    assert plan.source_state_ids == (4, 5)


def test_simulation_catchup_plan_skips_actions_when_endpoint_is_entry() -> None:
    action_1 = ActionSpec(action_id="ACTION1")

    plan = _simulation_catchup_plan(
        edges=(),
        entry_frame_hash="entry",
        simulated_end_frame_hash="entry",
        simulated_actions=(action_1,),
    )

    assert plan.actions == ()
    assert plan.source == "already_at_simulated_endpoint"


def test_simulation_catchup_plan_keeps_single_simulated_action() -> None:
    action_1 = ActionSpec(action_id="ACTION1")
    action_2 = ActionSpec(action_id="ACTION2")

    plan = _simulation_catchup_plan(
        edges=(
            _edge(
                source_state_id=1,
                successor_state_id=2,
                source_hash="entry",
                successor_hash="end",
                action=action_2,
            ),
        ),
        entry_frame_hash="entry",
        simulated_end_frame_hash="end",
        simulated_actions=(action_1,),
    )

    assert plan.actions == (action_1,)
    assert plan.source == "direct_simulated_path"


def test_known_state_action_match_normalizes_stored_game_action_repr() -> None:
    historical = ActionSpec(action_id="<GameAction.ACTION3: 3>")
    current = ActionSpec(action_id="ACTION3")
    edge = _edge(
        source_state_id=1,
        successor_state_id=2,
        source_hash="same",
        successor_hash="next",
        action=historical,
    )

    assert (
        _edge_for_action(
            (edge,),
            frame_hash="same",
            action=current,
            crop_edges=(0, 0, 0, 0),
        )
        is edge
    )


def test_known_state_action6_match_uses_target_value_and_current_bbox() -> None:
    historical = ActionSpec(
        action_id="ACTION6",
        data={"x": 24, "y": 24},
        target="old target",
        target_value=2,
    )
    current = ActionSpec(
        action_id="ACTION6",
        data={"x": 25, "y": 25},
        target="current target",
        target_value=2,
        target_bbox=(300, 300, 450, 450),
    )
    edge = _edge(
        source_state_id=1,
        successor_state_id=2,
        source_hash="same",
        successor_hash="next",
        action=historical,
    )

    assert (
        _edge_for_action(
            (edge,),
            frame_hash="same",
            action=current,
            crop_edges=(4, 4, 4, 4),
        )
        is edge
    )


def test_known_state_action6_match_rejects_different_target_value() -> None:
    historical = ActionSpec(
        action_id="ACTION6",
        data={"x": 24, "y": 24},
        target="old target",
        target_value=2,
    )
    current = ActionSpec(
        action_id="ACTION6",
        data={"x": 25, "y": 25},
        target="current target",
        target_value=3,
        target_bbox=(300, 300, 450, 450),
    )
    edge = _edge(
        source_state_id=1,
        successor_state_id=2,
        source_hash="same",
        successor_hash="next",
        action=historical,
    )

    assert (
        _edge_for_action(
            (edge,),
            frame_hash="same",
            action=current,
            crop_edges=(4, 4, 4, 4),
        )
        is None
    )


def test_action_history_changed_pixel_count_is_frame_area_percentage() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))

    entry = build_action_history_entry(
        frame_context=_frame_context(previous_frame),
        final_action=ActionSpec(action_id="ACTION1"),
        next_observation=Observation(id="current", step=1, frame=current_frame),
        change_summary="One pixel changed.",
        change_summary_crop_edges=None,
    )

    assert entry.changed_pixel_count == 0.0244


def test_summarize_change_skips_model_when_visible_frames_are_identical() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))
    session = _session(previous_frame, current_frame)
    change_model = _RecordingChangeModel()

    summarize_change(session, change_model=change_model, debug=DebugBus.disabled())

    assert change_model.calls == 0
    assert session.update_input is not None
    entry = session.update_input.action_history_entry
    assert entry is not None
    assert entry.changed_pixel_count == 0


def test_skipped_unchanged_change_summary_carries_previous_elements() -> None:
    previous_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    current_frame.putpixel((0, 0), (255, 0, 0))
    session = _session(previous_frame, current_frame)
    previous_element = ChangeSummaryElement(
        element_name="door",
        element_description="green door at the top edge",
        element_mutation="opened",
    )
    session.action_history.append(
        ActionHistoryEntry(
            action=ActionSpec(action_id="ACTION1"),
            controllable=True,
            changed_pixel_count=2.0,
            change_summary="- door: green door at the top edge; mutation: opened",
            change_elements=(previous_element,),
        )
    )
    change_model = _RecordingChangeModel()

    summarize_change(session, change_model=change_model, debug=DebugBus.disabled())

    assert change_model.calls == 0
    assert session.update_input is not None
    entry = session.update_input.action_history_entry
    assert entry is not None
    assert entry.change_elements == (
        ChangeSummaryElement(
            element_name="door",
            element_description="green door at the top edge",
            element_mutation="",
        ),
    )


def test_uncertain_change_summary_keeps_fresh_element_descriptions() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    current_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session = _session(previous_frame, current_frame)
    result = ChangeSummaryResult(
        elements=(
            ChangeSummaryElement(
                element_name="player",
                element_description="red square in the center",
                element_mutation="",
            ),
        ),
        change_detected=False,
        metadata={},
    )

    attach_change_summary(
        session,
        result=result,
        change_model=None,
        changed_pixel_count=1.0,
    )

    assert session.update_input is not None
    entry = session.update_input.action_history_entry
    assert entry is not None
    assert (
        "This action produced changes but it is uncertain what changed exactly."
        in entry.change_summary
    )
    assert "- player: red square in the center" in entry.change_summary


def test_uncertain_animation_summary_keeps_fresh_element_descriptions() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    mid_frame = Image.new("RGB", (8, 8), color=(0, 255, 0))
    current_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session = _session(previous_frame, current_frame)
    previous = session.current.observation
    current = session.next.observation
    session.update_input.frame_observations = (
        previous,
        Observation(id="mid", step=1, frame=mid_frame),
        current,
    )
    result = ChangeSummaryResult(
        elements=(
            ChangeSummaryElement(
                element_name="portal",
                element_description="purple portal on the right side",
                element_mutation="",
            ),
        ),
        change_detected=False,
        metadata={},
    )

    attach_change_summary(
        session,
        result=result,
        change_model=None,
        changed_pixel_count=1.0,
    )

    assert session.update_input is not None
    assert len(session.update_input.action_history_entries) == 2
    animation_entry = session.update_input.action_history_entries[1]
    assert (
        "animation produced changes but it is uncertain what changed exactly."
        in animation_entry.change_summary
    )
    assert "- portal: purple portal on the right side" in animation_entry.change_summary


def test_game_over_reset_clears_observed_transition_cursor() -> None:
    frame = Image.new("RGB", (64, 64), color=(0, 0, 0))
    previous = Observation(id="before-reset", step=3, frame=frame)
    previous_ref = ObservationRef(memory="state", id=previous.id)
    action = ActionSpec(action_id="ACTION1")
    trace = AgentTrace(
        step=previous.step,
        first_observation_ref=previous_ref,
        current_observation_ref=previous_ref,
        final_action=action,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_ResetEnvironment(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=previous,
        remaining_actions=1,
    )
    session.previous_observation = previous
    session.previous_observation_ref = previous_ref
    session.last_decision = DecisionResult(final_action=action, trace=trace)

    assert has_observed_transition(session)

    reset_after_game_over(session)

    assert session.previous_observation is None
    assert session.previous_observation_ref is None
    assert session.last_decision is None
    assert not has_observed_transition(session)


def test_level_completion_preserves_compacter_context_and_action_history() -> None:
    observation = _observation("level-complete")
    action = ActionSpec(action_id="ACTION1")
    history_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=10,
        change_summary="Completed the level.",
    )
    compacter_context = {
        "world_description": "buttons unlock doors",
        "action_effects": {"ACTION1": "moves"},
    }
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_CompletedLevelEnvironment(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=1,
    )
    session.queued_updater_actions = (action,)
    session.compacter_context = compacter_context
    session.level_action_count = 5
    session.last_submitted_level_action_count = 5
    session.action_history.append(history_entry)

    check_lifecycle(session)

    assert session.queued_updater_actions == ()
    assert session.level_action_count == 0
    assert session.last_submitted_level_action_count == 5
    assert session.compacter_context == compacter_context
    assert session.action_history == [history_entry]


def test_simulation_action_count_restarts_after_completed_level() -> None:
    action = ActionSpec(action_id="ACTION1")
    observation = _observation("current")
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        completed_levels=1,
    )
    session.action_history.append(
        ActionHistoryEntry(
            action=action,
            controllable=True,
            changed_pixel_count=1.0,
            change_summary="solved previous level",
            completed_levels=1,
            action_count=5,
        )
    )
    edge = _edge(
        source_state_id=1,
        successor_state_id=2,
        source_hash="a",
        successor_hash="b",
        action=action,
    )

    entries = _simulation_action_history_entries(session, edge)

    assert [entry.action_count for entry in entries] == [1]
    assert session.level_action_count == 1


def test_model_action_history_inputs_start_at_current_level_boundary() -> None:
    action = ActionSpec(action_id="ACTION1")
    observation = _observation("current-level-frame")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    contexts = ContextDocuments(
        agent=RoleContext(game=json.dumps({"current_strategy": "continue"}))
    )
    previous_level_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="solved old level",
        completed_levels=1,
        action_count=4,
    )
    current_level_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="explored new level",
        completed_levels=1,
        action_count=1,
    )
    latest_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="advanced in new level",
        completed_levels=1,
        action_count=2,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        first_observation_ref=observation_ref,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=4,
        observation=observation,
        observation_ref=observation_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=observation_ref,
    )
    session.action_history.extend((previous_level_entry, current_level_entry))
    session.compacter_action_history_start_index = 1
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=observation_ref,
        actual_next_observation_ref=observation_ref,
        decision_trace=_trace(observation),
        actual_next_observation=observation,
        action_history_entry=latest_entry,
    )
    compacter = _RecordingCompacter(
        previous_actions_summary="new level progress",
        previous_strategy_summary="new level strategy",
    )
    updater = _RecordingAgentUpdater(next_actions=(action,))

    run_compacter(
        session,
        compacter=compacter,
        state_memory=None,
        debug=DebugBus.disabled(),
    )
    run_updaters(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        state_memory=None,
        debug=DebugBus.disabled(),
    )

    expected_history = (current_level_entry, latest_entry)
    assert compacter.inputs[0].action_history == expected_history
    assert updater.inputs[0].action_history == expected_history


def test_full_history_buffer_compacts_before_updater_and_keeps_latest_raw() -> None:
    action = ActionSpec(action_id="ACTION1")
    observation = _observation("full-buffer-frame")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    contexts = ContextDocuments(
        agent=RoleContext(game=json.dumps({"current_strategy": "latest old plan"}))
    )
    first_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="first buffered action",
        completed_levels=0,
        action_count=1,
    )
    second_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="second buffered action",
        completed_levels=0,
        action_count=2,
    )
    latest_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="latest buffered action",
        completed_levels=0,
        action_count=3,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(updater_context_history_window=1),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        first_observation_ref=observation_ref,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=4,
        observation=observation,
        observation_ref=observation_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=observation_ref,
    )
    session.action_history.extend((first_entry, second_entry))
    session.strategy_history_buffer.extend(("first plan", "latest old plan"))
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=observation_ref,
        actual_next_observation_ref=observation_ref,
        decision_trace=_trace(observation),
        actual_next_observation=observation,
        action_history_entry=latest_entry,
    )
    compacter = _RecordingCompacter(
        previous_actions_summary="compacted older actions",
        previous_strategy_summary="compacted older strategies",
    )
    updater = _RecordingAgentUpdater(next_actions=(action,))

    run_compacter(
        session,
        compacter=compacter,
        state_memory=None,
        debug=DebugBus.disabled(),
    )
    run_updaters(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        state_memory=None,
        debug=DebugBus.disabled(),
    )

    assert compacter.inputs[0].action_history == (
        first_entry,
        second_entry,
        latest_entry,
    )
    assert compacter.inputs[0].strategy_history == (
        "first plan",
        "latest old plan",
    )
    assert updater.inputs[0].action_history == (latest_entry,)
    assert updater.inputs[0].previous_game_context_history == ("latest old plan",)
    assert "compacted older actions" in updater.inputs[0].previous_actions_summary
    assert session.compacter_action_history_start_index == 0
    assert session.strategy_history_buffer == [
        "latest old plan",
        '{\n  "current_strategy": "fresh plan"\n}',
    ]


def test_simulation_updater_action_history_starts_at_current_level_boundary(
    tmp_path,
) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action = ActionSpec(action_id="ACTION1")
    observation = _observation("simulation-current-level-frame")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    contexts = ContextDocuments(
        agent=RoleContext(game=json.dumps({"current_strategy": "simulate"}))
    )
    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=5,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=contexts,
    )
    previous_level_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="solved old level",
        completed_levels=1,
        action_count=4,
    )
    current_level_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="simulated current level",
        completed_levels=1,
        action_count=1,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        first_observation_ref=observation_ref,
    )
    session.action_history.extend((previous_level_entry, current_level_entry))
    session.compacter_action_history_start_index = 1
    compacter = _RecordingCompacter(
        previous_actions_summary="simulation current level",
        previous_strategy_summary="simulation strategy",
    )
    updater = _SequencedAgentUpdater((action,))

    result = _next_simulated_action(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        compacter=compacter,
        state_memory=memory,
        debug=DebugBus.disabled(),
        frame_context=FrameTurnContext(
            run_id="run-1",
            game_id="game-1",
            first_observation_ref=observation_ref,
            current_observation_ref=observation_ref,
            current_observation=observation,
            current_source_state_id=source.id,
            frame_index=0,
            frame_count=1,
            control_mode=FrameControlMode.real_environment_turn((action,)),
        ),
        current_action_item_count=1,
        turn_id=5,
    )

    assert result == action
    assert compacter.inputs[0].action_history == (current_level_entry,)
    assert updater.inputs[0].action_history == (current_level_entry,)


def test_simulation_strategy_history_matches_action_history_compaction(
    tmp_path,
) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action = ActionSpec(action_id="ACTION1")
    observation = _observation("simulation-compaction-frame")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    contexts = ContextDocuments(
        agent=RoleContext(game=json.dumps({"current_strategy": "simulate"}))
    )
    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=5,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=contexts,
    )
    first_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="older simulated action",
        completed_levels=0,
        action_count=1,
    )
    latest_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="latest simulated action",
        completed_levels=0,
        action_count=2,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(updater_context_history_window=1),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=10,
        first_observation_ref=observation_ref,
    )
    session.action_history.extend((first_entry, latest_entry))
    session.strategy_history_buffer.extend(("older strategy", "latest strategy"))
    compacter = _RecordingCompacter(
        previous_actions_summary="simulation older action",
        previous_strategy_summary="simulation older strategy",
    )
    updater = _SequencedAgentUpdater((action,))

    result = _next_simulated_action(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        compacter=compacter,
        state_memory=memory,
        debug=DebugBus.disabled(),
        frame_context=FrameTurnContext(
            run_id="run-1",
            game_id="game-1",
            first_observation_ref=observation_ref,
            current_observation_ref=observation_ref,
            current_observation=observation,
            current_source_state_id=source.id,
            frame_index=0,
            frame_count=1,
            control_mode=FrameControlMode.real_environment_turn((action,)),
        ),
        current_action_item_count=1,
        turn_id=5,
    )

    assert result == action
    assert compacter.inputs[0].action_history == (first_entry, latest_entry)
    assert compacter.inputs[0].strategy_history == (
        "older strategy",
        "latest strategy",
    )
    assert updater.inputs[0].action_history == (latest_entry,)
    assert updater.inputs[0].previous_game_context_history == ("latest strategy",)
    assert session.strategy_history_buffer == [
        "latest strategy",
        '{\n  "current_strategy": "simulated plan"\n}',
    ]


def test_level_completion_stores_final_compacter_summary(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments(
        agent=RoleContext(
            game=json.dumps(
                {
                    "current_strategy": "reach goal",
                }
            )
        )
    )
    first = _write_strategy_state(
        memory,
        contexts=contexts,
        step=1,
        current="reach switch",
    )
    second = _write_strategy_state(
        memory,
        contexts=contexts,
        step=2,
        current="walk to goal",
    )
    current_observation = _observation("solved-frame")
    current_source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        current_observation=current_observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=contexts,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(updater_context_history_window=1),
        game_id="game-1",
        latest_environment_observation=current_observation,
        remaining_actions=1,
        first_observation_ref=ObservationRef(memory="state", id="first"),
    )
    session.state_record_ids.extend((first.id, second.id))
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        observation=current_observation,
        observation_ref=ObservationRef(memory="state", id=current_observation.id),
        source_state_id=current_source.id,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=ObservationRef(memory="state", id="first"),
    )
    prior_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="switch opened path",
        completed_levels=0,
    )
    latest_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="goal reached",
        change_elements=(
            ChangeSummaryElement(
                element_name="goal",
                element_description="green goal tile beside the player",
                element_mutation="reached",
            ),
        ),
        completed_levels=1,
    )
    session.action_history.append(prior_entry)
    session.strategy_history_buffer.extend(
        (
            '{\n  "current_strategy": "reach switch"\n}',
            '{\n  "current_strategy": "walk to goal"\n}',
        )
    )
    session.compacter_context_summary = AgentCompacterSummary(
        world_description="previous world",
        special_events="previous event",
        action_effects={"ACTION1": "previous move"},
    )
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=ObservationRef(memory="state", id="previous"),
        actual_next_observation_ref=ObservationRef(
            memory="state",
            id=current_observation.id,
        ),
        decision_trace=_trace(current_observation),
        actual_next_observation=current_observation,
        action_history_entry=latest_entry,
    )
    compacter = _RecordingCompacter(
        previous_actions_summary="Use the switch, then goal.",
        previous_strategy_summary="Switch strategy solved the level.",
    )
    updater = _RecordingAgentUpdater(next_actions=(action,))

    run_compacter(
        session,
        compacter=compacter,
        state_memory=memory,
        debug=DebugBus.disabled(),
    )
    run_updaters(
        session,
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        state_memory=memory,
        debug=DebugBus.disabled(),
    )

    stored = memory.read_latest_compacter_level_summary(
        run_id="run-1",
        game_id="game-1",
    )
    assert stored is not None
    assert stored.completed_level == 1
    assert stored.source_state_ids == (current_source.id,)
    assert stored.previous_actions_summary == "Use the switch, then goal."
    assert stored.previous_strategy_summary == "Switch strategy solved the level."
    assert session.compacter_action_history_start_index == 2
    assert compacter.inputs[0].action_history == (prior_entry, latest_entry)
    assert compacter.inputs[0].strategy_history == (
        '{\n  "current_strategy": "reach switch"\n}',
        '{\n  "current_strategy": "walk to goal"\n}',
    )
    assert "world_description: world" in updater.inputs[0].world_model_context
    assert "- ACTION1: moves" in updater.inputs[0].world_model_context
    assert "Use the switch, then goal." in updater.inputs[0].previous_actions_summary
    assert (
        updater.inputs[0].previous_strategy_summary
        == "Switch strategy solved the level."
    )
    assert updater.inputs[0].action_history == ()
    assert updater.inputs[0].previous_game_context_history == ()
    assert updater.inputs[0].reset_notice == LEVEL_SOLVED_RESET_NOTICE


def test_post_reset_bootstrap_ignores_pre_reset_controllable_actions() -> None:
    observation = _observation("after-reset")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    action = ActionSpec(action_id="ACTION1")
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=observation,
        remaining_actions=1,
        game_start_turn_id=6,
        game_start_reason="game_over_reset",
    )
    session.action_history.extend(
        (
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=0,
                change_summary="pre-reset action",
            ),
            ActionHistoryResetMarker(reason="game_over_reset", restart_count=1),
        )
    )
    session.first_observation_ref = observation_ref
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=6,
        observation=observation,
        observation_ref=observation_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=observation_ref,
    )
    updater = _RecordingAgentUpdater(next_actions=(action,))

    bootstrap_agent_updater_decision(
        session,
        contexts=ContextDocuments(),
        compacter=_RecordingCompacter(),
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        debug=DebugBus.disabled(),
    )

    assert session.queued_updater_actions == (action,)
    assert updater.calls == 1
    assert updater.inputs[0].action_history == ()


def test_agent_updater_receives_strategy_history_and_updates_context() -> None:
    frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
    action = ActionSpec(action_id="ACTION1")
    contexts = ContextDocuments()
    contexts.agent.game = json.dumps(
        {
            "current_strategy": "reach the bright area",
        }
    )
    updater = _RecordingAgentUpdater(next_actions=(action,))

    apply_agent_context_update(
        contexts=contexts,
        updater_tasks=UpdaterTaskRegistry(agent_updater=updater),
        debug=DebugBus.disabled(),
        frame_context=_frame_context(frame),
        current_observation=Observation(id="current", step=1, frame=frame),
        action_history=(),
        allowed_action_source=(action,),
        previous_game_context_history=("prior context",),
        compacter_context=None,
        turn_id=1,
    )

    assert updater.inputs[0].previous_game_context_history == ("prior context",)
    assert json.loads(contexts.agent.game) == {
        "current_strategy": "fresh plan",
    }


def test_change_summary_receives_previous_change_elements() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    current_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session = _session(previous_frame, current_frame)
    previous_element = ChangeSummaryElement(
        element_name="player",
        element_description="red square",
        element_mutation="moved right",
    )
    session.action_history.append(
        ActionHistoryEntry(
            action=ActionSpec(action_id="ACTION1"),
            controllable=True,
            changed_pixel_count=1,
            change_summary="- player: red square; mutations: moved right",
            change_elements=(previous_element,),
        )
    )
    change_model = _RecordingChangeModel()

    summarize_change_model(session, change_model=change_model, debug=DebugBus.disabled())

    assert change_model.previous_change_elements == [(previous_element,)]


def test_level_completion_change_summary_omits_final_new_level_bundle_frame() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    solved_frame = Image.new("RGB", (8, 8), color=(0, 255, 0))
    new_level_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    next_observation = Observation(
        id="level-complete",
        step=1,
        frame=solved_frame,
        frames=(solved_frame, new_level_frame),
        metadata={"levels_completed": 1},
    )
    session = _resolved_level_completion_session(previous_frame, next_observation)
    change_model = _RecordingChangeModel()

    summarize_change_model(session, change_model=change_model, debug=DebugBus.disabled())

    assert change_model.frame_observations[0] is not None
    assert [observation.id for observation in change_model.frame_observations[0]] == [
        "previous",
        "level-complete-frame-0",
    ]


def test_level_completion_change_summary_keeps_single_frame_when_trimmed_to_one() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    new_level_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    next_observation = Observation(
        id="level-complete",
        step=1,
        frame=new_level_frame,
        frames=(new_level_frame,),
        metadata={"levels_completed": 1},
    )
    session = _resolved_level_completion_session(previous_frame, next_observation)
    change_model = _RecordingChangeModel()

    summarize_change_model(session, change_model=change_model, debug=DebugBus.disabled())

    assert change_model.frame_observations[0] is not None
    assert [observation.id for observation in change_model.frame_observations[0]] == [
        "previous",
    ]


def test_level_completion_compacter_uses_trimmed_previous_level_frame() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    solved_frame = Image.new("RGB", (8, 8), color=(0, 255, 0))
    new_level_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    next_observation = Observation(
        id="level-complete",
        step=1,
        frame=solved_frame,
        frames=(solved_frame, new_level_frame),
        metadata={"levels_completed": 1},
    )
    session = _resolved_level_completion_session(previous_frame, next_observation)
    assert session.update_input is not None
    session.update_input.action_history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="level solved",
        completed_levels=1,
    )
    compacter = _RecordingCompacter()

    run_compacter(
        session,
        compacter=compacter,
        state_memory=None,
        debug=DebugBus.disabled(),
    )
    run_updaters(
        session,
        contexts=ContextDocuments(agent=RoleContext(game="{}")),
        updater_tasks=UpdaterTaskRegistry(
            agent_updater=_RecordingAgentUpdater(
                next_actions=(ActionSpec(action_id="ACTION1"),),
            )
        ),
        state_memory=None,
        debug=DebugBus.disabled(),
    )

    assert compacter.inputs[0].current_observation.id == "level-complete-frame-0"
    assert session.update_input.actual_next_observation.id == "level-complete-frame-1"


def test_level_completion_compacter_uses_previous_frame_when_only_new_level_frame() -> None:
    previous_frame = Image.new("RGB", (8, 8), color=(0, 0, 0))
    new_level_frame = Image.new("RGB", (8, 8), color=(255, 0, 0))
    next_observation = Observation(
        id="level-complete",
        step=1,
        frame=new_level_frame,
        frames=(new_level_frame,),
        metadata={"levels_completed": 1},
    )
    session = _resolved_level_completion_session(previous_frame, next_observation)
    assert session.update_input is not None
    session.update_input.action_history_entry = ActionHistoryEntry(
        action=ActionSpec(action_id="ACTION1"),
        controllable=True,
        changed_pixel_count=1.0,
        change_summary="level solved",
        completed_levels=1,
    )
    compacter = _RecordingCompacter()

    run_compacter(
        session,
        compacter=compacter,
        state_memory=None,
        debug=DebugBus.disabled(),
    )
    run_updaters(
        session,
        contexts=ContextDocuments(agent=RoleContext(game="{}")),
        updater_tasks=UpdaterTaskRegistry(
            agent_updater=_RecordingAgentUpdater(
                next_actions=(ActionSpec(action_id="ACTION1"),),
            )
        ),
        state_memory=None,
        debug=DebugBus.disabled(),
    )

    assert compacter.inputs[0].current_observation.id == "previous"
    assert session.update_input.actual_next_observation.id == "level-complete"


def _write_strategy_state(
    memory: StateMemory,
    *,
    contexts: ContextDocuments,
    step: int,
    current: str,
):
    observation = _observation(f"obs-{step}")
    return memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=step,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=ActionSpec(action_id=f"ACTION{step}"),
        contexts=contexts,
        agent_trace=_trace(observation),
        metadata={
            "agent_context_history": {
                "current_strategy": current,
            }
        },
    )


class _FakeEnvironment:
    def __init__(self, observations: tuple[Observation, ...]) -> None:
        self.observations = list(observations)
        self.submitted_actions: list[ActionSpec] = []

    def step(self, action: ActionSpec) -> Observation:
        self.submitted_actions.append(action)
        if not self.observations:
            raise AssertionError("fake environment has no queued observation")
        return self.observations.pop(0)


def _edge(
    *,
    source_state_id: int,
    successor_state_id: int,
    source_hash: str,
    successor_hash: str,
    action: ActionSpec,
) -> KnownStateTransitionEdge:
    return KnownStateTransitionEdge(
        source_state_id=source_state_id,
        successor_state_id=successor_state_id,
        source_frame_hash=source_hash,
        successor_frame_hash=successor_hash,
        action=action,
        successor_observation=_observation(f"obs-{successor_state_id}"),
        action_history_entries=(
            ActionHistoryEntry(
                action=action,
                controllable=True,
                changed_pixel_count=1.0,
                change_summary=f"edge {source_state_id}",
            ),
        ),
    )


def _frame_context(frame: Image.Image) -> FrameTurnContext:
    observation = Observation(id="previous", step=0, frame=frame)
    ref = ObservationRef(memory="state", id=observation.id)
    action = ActionSpec(action_id="ACTION1")
    return FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )


def _session(
    previous_frame: Image.Image,
    current_frame: Image.Image,
    *,
    source_state_id: int | None = None,
) -> GameLoopSession:
    action = ActionSpec(action_id="ACTION1")
    previous = Observation(id="previous", step=0, frame=previous_frame)
    current = Observation(id="current", step=1, frame=current_frame)
    previous_ref = ObservationRef(memory="state", id=previous.id)
    current_ref = ObservationRef(memory="state", id=current.id)
    control_mode = FrameControlMode.real_environment_turn((action,))
    trace = AgentTrace(
        step=previous.step,
        first_observation_ref=previous_ref,
        current_observation_ref=previous_ref,
        final_action=action,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=object(),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=previous,
        remaining_actions=1,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=previous,
        observation_ref=previous_ref,
        source_state_id=source_state_id,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=previous_ref,
    )
    session.next = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=current,
        observation_ref=current_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        first_observation_ref=previous_ref,
    )
    session.decision = DecisionResult(final_action=action, trace=trace)
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=previous_ref,
        actual_next_observation_ref=current_ref,
        decision_trace=trace,
        actual_next_observation=current,
    )
    return session


def _resolved_level_completion_session(
    previous_frame: Image.Image,
    next_observation: Observation,
) -> GameLoopSession:
    action = ActionSpec(action_id="ACTION1")
    previous = Observation(id="previous", step=0, frame=previous_frame)
    previous_ref = ObservationRef(memory="state", id=previous.id)
    trace = AgentTrace(
        step=previous.step,
        first_observation_ref=previous_ref,
        current_observation_ref=previous_ref,
        final_action=action,
    )
    session = GameLoopSession(
        config=RuntimeConfig(run_id="run-1"),
        environment=_FakeEnvironment((next_observation,)),
        environment_config=EnvironmentConfig(),
        game_id="game-1",
        latest_environment_observation=previous,
        remaining_actions=1,
        completed_levels=0,
    )
    session.current = FrameTurnSnapshot(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        observation=previous,
        observation_ref=previous_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=previous_ref,
    )
    session.decision = DecisionResult(final_action=action, trace=trace)
    resolve_next_snapshot(
        session,
        debug=DebugBus.disabled(),
        change_model=_RecordingChangeModel(),
    )
    session.previous_observation = previous
    session.previous_observation_ref = previous_ref
    session.last_decision = session.decision
    next_snapshot = session.next
    assert next_snapshot is not None
    session.current = FrameTurnSnapshot(
        run_id=next_snapshot.run_id,
        game_id=next_snapshot.game_id,
        turn_id=next_snapshot.turn_id,
        observation=next_snapshot.observation,
        observation_ref=next_snapshot.observation_ref,
        source_state_id=next_snapshot.source_state_id,
        frame_index=next_snapshot.frame_index,
        frame_count=next_snapshot.frame_count,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        first_observation_ref=next_snapshot.first_observation_ref,
        previous_observation_ref=next_snapshot.previous_observation_ref,
        recent_action_history=next_snapshot.recent_action_history,
    )
    prepare_observed_transition(session)
    return session


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(1, 2, 3)),
    )


def _colored_observation(
    observation_id: str,
    color: tuple[int, int, int],
) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=color),
    )


def _trace(observation: Observation) -> AgentTrace:
    ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=ActionSpec(action_id="ACTION1"),
    )


class _RecordingChangeModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"input_image_crop_arc_grid_edges": 4})()
        self.calls = 0
        self.previous_change_elements: list[tuple[ChangeSummaryElement, ...]] = []
        self.frame_observations: list[tuple[Observation, ...] | None] = []

    def summarize(self, *args: object, **kwargs: object) -> ChangeSummaryResult:
        self.calls += 1
        self.previous_change_elements.append(tuple(kwargs["previous_change_elements"]))
        self.frame_observations.append(kwargs.get("frame_observations"))
        return ChangeSummaryResult(
            elements=(
                ChangeSummaryElement(
                    element_name="element",
                    element_description="visible object",
                    element_mutation="model summary",
                ),
            ),
            change_detected=True,
            metadata={},
        )


class _RecordingAgentUpdater:
    def __init__(self, *, next_actions: tuple[ActionSpec, ...]) -> None:
        self.next_actions = next_actions
        self.calls = 0
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        self.calls += 1
        self.inputs.append(update_input)
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {
                    "current_strategy": "fresh plan",
                }
            ),
            next_actions=self.next_actions,
        )


class _SequencedAgentUpdater:
    def __init__(self, actions: tuple[ActionSpec, ...]) -> None:
        self.actions = list(actions)
        self.inputs: list[AgentGameContextUpdateInput] = []

    def update_agent_context(
        self,
        update_input: AgentGameContextUpdateInput,
    ) -> AgentGameContextUpdateResult:
        if not self.actions:
            raise AssertionError("no sequenced updater action left")
        self.inputs.append(update_input)
        action = self.actions.pop(0)
        return AgentGameContextUpdateResult(
            context=json.dumps(
                {
                    "current_strategy": "simulated plan",
                }
            ),
            next_actions=(action,),
        )


class _RecordingCompacter:
    config = None

    def __init__(
        self,
        *,
        previous_actions_summary: str = "actions",
        previous_strategy_summary: str = "strategies",
    ) -> None:
        self.inputs: list[AgentCompacterInput] = []
        self.previous_actions_summary = previous_actions_summary
        self.previous_strategy_summary = previous_strategy_summary

    def compact_agent_context(
        self,
        compacter_input: AgentCompacterInput,
    ) -> AgentCompacterSummary:
        self.inputs.append(compacter_input)
        self._model_input_debug_records = [
            {
                "call_slot": "compacter",
                "provider": "fake",
                "model": "fake-compacter",
                "phase": "compact_agent_context",
                "attempt": 0,
                "request": {"turn": len(self.inputs)},
                "usage": None,
                "metadata": {},
            }
        ]
        return AgentCompacterSummary(
            world_description="world",
            special_events="none",
            action_effects={"ACTION1": "moves"},
            previous_actions_summary=self.previous_actions_summary,
            previous_strategy_summary=self.previous_strategy_summary,
        )


class _ResetEnvironment:
    def reset(self) -> Observation:
        frame = Image.new("RGB", (64, 64), color=(1, 1, 1))
        return Observation(id="after-reset", step=0, frame=frame)

    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id="game-1")


class _CompletedLevelEnvironment:
    def get_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(game_id="game-1", levels_completed=1)
