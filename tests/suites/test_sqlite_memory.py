"""Smoke tests for SQLite memory domains."""

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
    TurnMetrics,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory


def _trace(observation_ref: ObservationRef, action: ActionSpec) -> AgentTrace:
    return AgentTrace(
        step=0,
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
        m_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(m_states)")
        }

    assert {
        "m_states",
        "e_experiments",
        "model_input_debug_records",
        "run_metadata",
    }.issubset(tables)
    assert "agent_context_json" in m_columns
    assert "turn_metrics_json" in m_columns
    assert "world_context_json" not in m_columns
    assert state.read_latest_state("game-1") is None
    assert experimental.list_experiments() == []


def test_state_memory_writes_reads_and_cleans_agent_states(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)

    first = state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="old")),
        agent_trace=_trace(observation_ref, action),
    )
    second = state.write_state(
        run_id="run-2",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=Observation(id="obs-1", step=1, frame={"frame": 1}),
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(general="K", game="new")),
        agent_trace=_trace(observation_ref, action),
        turn_metrics=TurnMetrics(time_cost=2.0, trace_cost=0.5),
    )
    state.write_state(
        run_id="run-1",
        game_id="game-2",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="other")),
        agent_trace=_trace(observation_ref, action),
    )

    latest = state.read_latest_state("game-1")
    assert latest is not None
    assert latest.id == second.id
    assert latest.agent_context == RoleContext(general="K", game="new")
    assert latest.turn_metrics.time_cost == 2.0
    assert first.turn_metrics == TurnMetrics()

    state.cleanup_keep_latest_per_game()

    remaining = state.list_states()
    assert len(remaining) == 2
    assert first.id not in {record.id for record in remaining}

    state.clear_states()
    assert state.list_states() == []


def test_state_memory_reads_latest_agent_general_context(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)

    assert state.read_latest_general_contexts() == ContextDocuments()

    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(general="agent K 1")),
        agent_trace=_trace(observation_ref, action),
    )
    state.write_state(
        run_id="run-2",
        game_id="game-2",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(general="agent K 2")),
        agent_trace=_trace(observation_ref, action),
    )

    contexts = state.read_latest_general_contexts()
    assert contexts.agent == RoleContext(general="agent K 2", game="")


def test_experimental_memory_writes_reads_and_resolves_output_frames(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(
        tool="world",
        source_state_id=1,
        action=ActionSpec(action_id="ACTION1"),
    )
    result = ToolResult(
        id="world-result-1",
        tool="world",
        output={"frame": "predicted"},
        source_observation_ref=source_ref,
        source_state_id=1,
        action=ActionSpec(action_id="ACTION1"),
        metadata={"quality": "fake"},
    )

    record = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=4,
        tool_call=call,
        output_description=Observation(
            id=result.id,
            step=2,
            frame=result.output,
            frames=(result.output,),
        ),
        tool_result=result,
    )

    latest = experimental.read_experiment(record.id)
    assert latest is not None
    assert latest.source_state_id == 1
    assert latest.tool_name == "world"
    assert latest.output_description["frame"] == {"frame": "predicted"}
    assert latest.tool_result["metadata"] == {"quality": "fake"}


def test_memory_serializes_and_rehydrates_visual_frames_at_64x64(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-image")
    action = ActionSpec(action_id="ACTION1")
    trace = _trace(source_ref, action)
    source = Observation(
        id="obs-image",
        step=0,
        frame=Image.new("RGB", (20, 30), color=(1, 2, 3)),
    )

    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=source,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=trace,
    )
    latest = state.read_latest_state("game-1")

    assert latest is not None
    assert isinstance(latest.current_observation["frame"], Image.Image)
    assert latest.current_observation["frame"].size == (64, 64)

    result = ToolResult(
        id="world-image",
        tool="world",
        output=Image.new("RGB", (11, 13), color=(9, 8, 7)),
        source_observation_ref=source_ref,
        source_state_id=1,
        action=action,
    )
    experiment = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="world", source_state_id=1, action=action),
        output_description=Observation(id=result.id, step=0, frame=result.output),
        tool_result=result,
    )
    stored = experimental.read_experiment(experiment.id)

    assert stored is not None
    assert isinstance(stored.output_description["frame"], Image.Image)
    assert stored.output_description["frame"].size == (64, 64)


def test_state_memory_clears_current_memory_tables(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=_trace(observation_ref, action),
    )
    experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="goal", source_state_id=1),
        output_description=Observation(id="goal-0", step=0, frame={"goal": True}),
        tool_result=ToolResult(
            id="goal-0",
            tool="goal",
            output={"goal": True},
            source_observation_ref=observation_ref,
            source_state_id=1,
        ),
    )

    state.clear_memory_tables()

    assert state.list_states() == []
    assert experimental.list_experiments() == []
