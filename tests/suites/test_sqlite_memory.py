"""Smoke tests for current SQLite memory domains."""

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
    TurnMetrics,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory
from face_of_agi.models.memory import GameMemoryDocument


def _trace(observation: Observation, action: ActionSpec) -> AgentTrace:
    observation_ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )


def test_sqlite_initializes_current_memory_tables(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)

    with sqlite3.connect(database.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "m_states",
        "e_experiments",
        "run_metadata",
        "model_input_debug_records",
        "model_call_events",
        "environment_step_events",
    }.issubset(tables)
    assert state.read_latest_state("game-1") is None
    assert experimental.list_experiments(run_id="run-1") == []


def test_state_memory_writes_model_and_environment_timing_events(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)

    model_event = state.write_model_call_event(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        role="change",
        provider="vllm",
        model="model",
        event="provider_end",
        status="success",
        queue_wait_seconds=1.25,
        duration_seconds=2.5,
        timeout_seconds=90.0,
        metadata={"phase": "complete"},
    )
    environment_event = state.write_environment_step_event(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        step=7,
        action={"action_id": "ACTION1"},
        status="success",
        duration_seconds=0.125,
        remaining_actions=9,
        metadata={"next_observation_id": "obs-8"},
    )

    assert model_event.role == "change"
    assert model_event.queue_wait_seconds == 1.25
    assert model_event.metadata["phase"] == "complete"
    assert state.list_model_call_events(run_id="run-1")[0].id == model_event.id

    assert environment_event.action == {"action_id": "ACTION1"}
    assert environment_event.remaining_actions == 9
    assert (
        state.list_environment_step_events(game_id="game-1")[0].id
        == environment_event.id
    )


def test_state_memory_writes_reads_and_cleans_m_states(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    action = ActionSpec(action_id="ACTION1")

    first_observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    first = state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=first_observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="old")),
        agent_trace=_trace(first_observation, action),
    )
    second_observation = Observation(id="obs-1", step=1, frame={"frame": 1})
    second = state.write_state(
        run_id="run-2",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=second_observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="new")),
        agent_trace=_trace(second_observation, action),
        turn_metrics=TurnMetrics(trace_cost=1.5, cumulative_score=2.0),
    )
    other_observation = Observation(id="obs-2", step=0, frame={"frame": 2})
    state.write_state(
        run_id="run-1",
        game_id="game-2",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=other_observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="other")),
        agent_trace=_trace(other_observation, action),
    )

    latest = state.read_latest_state("game-1")
    assert latest is not None
    assert latest.id == second.id
    assert latest.agent_context.game == "new"
    assert latest.turn_metrics.trace_cost == 1.5
    assert latest.turn_metrics.cumulative_score == 2.0
    assert first.agent_context.game == "old"
    assert state.read_latest_state("never-played") is None

    state.cleanup_keep_latest_per_game()

    remaining = state.list_states()
    assert len(remaining) == 2
    assert first.id not in {record.id for record in remaining}

    state.clear_states()
    assert state.list_states() == []


def test_state_memory_persists_action6_target_value_without_bbox(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(
        action_id="ACTION6",
        data={"x": 4, "y": 5},
        target="blue center tile",
        target_value=9,
        target_bbox=(200, 300, 400, 500),
    )

    source = state.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=ContextDocuments(agent=RoleContext(game="before")),
    )
    completed = state.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="after")),
        agent_trace=_trace(observation, action),
        action_history_entry=ActionHistoryEntry(
            action=action,
            controllable=True,
            changed_pixel_count=1,
            change_summary="blue changed",
        ),
    )
    history_action = completed.metadata["action_history_entry"]["action"]

    assert completed.chosen_action == {
        "action_id": "ACTION6",
        "data": {"x": 4, "y": 5},
        "target": "blue center tile",
        "target_value": 9,
    }
    assert "target_bbox" not in completed.chosen_action
    assert completed.agent_trace["final_action"]["target_value"] == 9
    assert "target_bbox" not in completed.agent_trace["final_action"]
    assert history_action["target_value"] == 9
    assert "target_bbox" not in history_action


def test_state_memory_completes_prewritten_frame_turn_with_game_memory(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")

    source = state.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        contexts=ContextDocuments(agent=RoleContext(game="before")),
    )
    completed = state.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="after")),
        agent_trace=_trace(observation, action),
        game_memory=GameMemoryDocument(
            "memory text",
            metadata={"available": True, "memory_char_count": 11},
        ),
        game_memory_updated_this_turn=True,
    )

    assert completed.chosen_action == {
        "action_id": "ACTION1",
        "data": None,
        "target": None,
        "target_value": None,
    }
    assert completed.metadata["game_memory"]["document"] == "memory text"
    assert completed.metadata["game_memory"]["updated_this_turn"] is True


def test_state_memory_reads_latest_general_contexts_across_games(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    action = ActionSpec(action_id="ACTION1")

    assert state.read_latest_general_contexts() == ContextDocuments()

    for run_id, game_id, general in (
        ("run-1", "game-1", "agent K 1"),
        ("run-2", "game-2", "agent K 2"),
    ):
        observation = Observation(id=f"obs-{game_id}", step=0, frame={"frame": 0})
        state.write_state(
            run_id=run_id,
            game_id=game_id,
            step=0,
            frame_index=0,
            frame_count=1,
            current_observation=observation,
            chosen_action=action,
            contexts=ContextDocuments(
                agent=RoleContext(general=general, game=f"L {game_id}")
            ),
            agent_trace=_trace(observation, action),
        )

    contexts = state.read_latest_general_contexts()
    assert contexts.agent == RoleContext(general="agent K 2", game="")


def test_experimental_memory_writes_reads_and_resolves_output_frames(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(
        tool="world",
        source_state_id=7,
        action=ActionSpec(action_id="ACTION1"),
    )
    output = Observation(
        id="world-result-1",
        step=2,
        frame=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )
    result = ToolResult(
        id="world-result-1",
        tool="world",
        output={"description": "predicted"},
        source_observation_ref=source_ref,
        source_state_id=7,
        action=ActionSpec(action_id="ACTION1"),
        metadata={"quality": "fake"},
    )

    record = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=4,
        tool_call=call,
        output_description=output,
        tool_result=result,
    )

    assert record.tool_name == "world"
    assert record.source_state_id == 7
    assert experimental.read_experiment(record.id).tool_result["id"] == "world-result-1"
    assert experimental.list_experiments(run_id="run-1")[0].id == record.id

    experimental.clear_experiments()
    assert experimental.list_experiments(run_id="run-1") == []
