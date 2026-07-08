"""Smoke tests for SQLite memory domains."""

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    Observation,
    ObservationRef,
    PostDecisionPredictions,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory


def test_sqlite_initializes_separate_memory_tables(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    experimental_record = experimental.write_record(
        run_id="run-1",
        game_id="game-1",
        step=0,
        kind="tool.world",
        payload={"id": "tool-0"},
    )

    with sqlite3.connect(database.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "state_records",
        "experimental_records",
        "m_states",
        "e_experiments",
    }.issubset(tables)
    assert experimental_record.domain == "experimental"
    assert state.read_latest_state("game-1") is None
    assert experimental.list_records(run_id="run-1")[0].payload == {"id": "tool-0"}

    with sqlite3.connect(database.path) as connection:
        e_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(e_experiments)")
        }
    assert "source_observation_ref_json" in e_columns
    assert "output_observation_json" in e_columns
    assert "input_observation_json" not in e_columns
    assert "step" not in e_columns
    assert "frame_index" not in e_columns
    assert "frame_count" not in e_columns
    with sqlite3.connect(database.path) as connection:
        m_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(m_states)")
        }
    assert "world_prediction_json" in m_columns
    assert "goal_prediction_json" in m_columns


def test_sqlite_adds_prediction_columns_to_existing_m_states(tmp_path) -> None:
    database_path = tmp_path / "memory.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE m_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step INTEGER,
                frame_index INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                current_observation_json TEXT NOT NULL,
                chosen_action_json TEXT NOT NULL,
                world_context_json TEXT NOT NULL,
                goal_context_json TEXT NOT NULL,
                agent_context_json TEXT NOT NULL,
                agent_trace_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    StateMemory(SQLiteDatabase(database_path))

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(m_states)")
        }
    assert "world_prediction_json" in columns
    assert "goal_prediction_json" in columns


def test_state_memory_writes_reads_and_cleans_m_states(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    predictions = PostDecisionPredictions(
        world_prediction=ToolResult(
            id="world-post",
            tool="world",
            predicted_observation={"frame": "world"},
            source_observation_ref=observation_ref,
            action=action,
        ),
        goal_prediction=ToolResult(
            id="goal-post",
            tool="goal",
            predicted_observation={"frame": "goal"},
            source_observation_ref=observation_ref,
        ),
    )

    first = state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="old")),
        agent_trace=trace,
        post_decision_predictions=predictions,
    )
    second = state.write_state(
        run_id="run-2",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=Observation(id="obs-1", step=1, frame={"frame": 1}),
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="new")),
        agent_trace=trace,
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
        agent_trace=trace,
    )

    latest = state.read_latest_state("game-1")
    assert latest is not None
    assert latest.id == second.id
    assert latest.agent_context.game == "new"
    assert latest.agent_trace["tool_calls"] == []
    assert latest.world_prediction is None
    assert latest.goal_prediction is None
    assert first.world_prediction["id"] == "world-post"
    assert first.goal_prediction["id"] == "goal-post"
    assert state.read_latest_state("never-played") is None

    state.cleanup_keep_latest_per_game()

    remaining = state.list_states()
    assert len(remaining) == 2
    assert first.id not in {record.id for record in remaining}

    state.clear_states()
    assert state.list_states() == []


def test_experimental_memory_writes_reads_and_resolves_output_frames(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(
        tool="world",
        observation_ref=source_ref,
        action=ActionSpec(action_id="ACTION1"),
    )
    result = ToolResult(
        id="world-result-1",
        tool="world",
        predicted_observation={"frame": "predicted"},
        source_observation_ref=source_ref,
        action=ActionSpec(action_id="ACTION1"),
        metadata={"quality": "fake"},
    )

    record = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=4,
        tool_call=call,
        output_observation=Observation(
            id=result.id,
            step=2,
            frame=result.predicted_observation,
            frames=(result.predicted_observation,),
        ),
        tool_result=result,
    )

    latest = experimental.read_experiment(record.id)
    assert latest is not None
    assert latest.source_observation_ref == source_ref
    assert latest.tool_name == "world"
    assert latest.output_observation["frame"] == {"frame": "predicted"}
    assert latest.tool_result["metadata"] == {"quality": "fake"}

    resolved = experimental.resolve_observation(
        ObservationRef(memory="experimental", id=str(record.id))
    )
    assert resolved is not None
    assert resolved.frame == {"frame": "predicted"}


def test_memory_serializes_and_rehydrates_visual_frames_at_64x64(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-image")
    action = ActionSpec(action_id="ACTION1")
    trace = AgentTrace(
        step=0,
        first_observation_ref=source_ref,
        current_observation_ref=source_ref,
        final_action=action,
    )
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
        predicted_observation=Image.new("RGB", (11, 13), color=(9, 8, 7)),
        source_observation_ref=source_ref,
        action=action,
    )
    experiment = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="world", observation_ref=source_ref, action=action),
        output_observation=Observation(
            id=result.id,
            step=0,
            frame=result.predicted_observation,
        ),
        tool_result=result,
    )
    stored = experimental.read_experiment(experiment.id)
    resolved = experimental.resolve_observation(
        ObservationRef(memory="experimental", id=str(experiment.id))
    )

    assert stored is not None
    assert isinstance(stored.output_observation["frame"], Image.Image)
    assert stored.output_observation["frame"].size == (64, 64)
    assert resolved is not None
    assert isinstance(resolved.frame, Image.Image)
    assert resolved.frame.size == (64, 64)


def test_experimental_memory_cleanup_keeps_latest_turns_per_run_and_game(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(tool="goal", observation_ref=source_ref)
    result = ToolResult(
        id="goal-result",
        tool="goal",
        predicted_observation={"frame": "goal"},
        source_observation_ref=source_ref,
    )

    for run_id, game_id, turns in (
        ("run-1", "game-1", (1, 2, 3)),
        ("run-1", "game-2", (1, 2, 3)),
        ("run-2", "game-1", (1, 2, 3)),
    ):
        for turn_id in turns:
            experimental.write_experiment(
                run_id=run_id,
                game_id=game_id,
                turn_id=turn_id,
                tool_call=call,
                output_observation=Observation(
                    id=f"goal-{run_id}-{game_id}-{turn_id}",
                    step=turn_id,
                    frame={"turn": turn_id},
                ),
                tool_result=result,
            )

    experimental.cleanup_keep_latest_turns_per_game(
        run_id="run-1",
        game_id="game-1",
        max_turns=2,
    )

    run_1_game_1_turns = {
        record.turn_id
        for record in experimental.list_experiments(
            run_id="run-1",
            game_id="game-1",
        )
    }
    run_1_game_2_turns = {
        record.turn_id
        for record in experimental.list_experiments(
            run_id="run-1",
            game_id="game-2",
        )
    }
    run_2_game_1_turns = {
        record.turn_id
        for record in experimental.list_experiments(
            run_id="run-2",
            game_id="game-1",
        )
    }

    assert run_1_game_1_turns == {2, 3}
    assert run_1_game_2_turns == {1, 2, 3}
    assert run_2_game_1_turns == {1, 2, 3}


def test_state_memory_clears_current_and_legacy_memory_tables(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )
    state.write_record(
        run_id="run-1",
        game_id="game-1",
        step=0,
        kind="legacy",
        payload={"old": True},
    )
    experimental.write_record(
        run_id="run-1",
        game_id="game-1",
        step=0,
        kind="legacy",
        payload={"old": True},
    )
    state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(),
        agent_trace=trace,
    )
    experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="goal", observation_ref=observation_ref),
        output_observation=Observation(id="goal-0", step=0, frame={"goal": True}),
        tool_result=ToolResult(
            id="goal-0",
            tool="goal",
            predicted_observation={"goal": True},
            source_observation_ref=observation_ref,
        ),
    )

    state.clear_memory_tables()

    assert state.list_states() == []
    assert state.list_records() == []
    assert experimental.list_records() == []
    assert experimental.list_experiments() == []
