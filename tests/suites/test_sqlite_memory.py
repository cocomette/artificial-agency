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
    TurnMetrics,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.memory import ExperimentalMemory, SQLiteDatabase, StateMemory


def test_sqlite_initializes_separate_memory_tables(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    experimental = ExperimentalMemory(database)
    observation_ref = ObservationRef(memory="state", id="obs-0")
    experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="world", source_state_id=1),
        output_description=Observation(id="tool-0", step=0, frame={"id": "tool-0"}),
        tool_result=ToolResult(
            id="tool-0",
            tool="world",
            predicted_description={"id": "tool-0"},
            source_observation_ref=observation_ref,
        ),
    )

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
        "model_input_debug_records",
    }.issubset(tables)
    assert state.read_latest_state("game-1") is None
    assert experimental.list_experiments(run_id="run-1")[0].tool_result["id"] == (
        "tool-0"
    )

    with sqlite3.connect(database.path) as connection:
        e_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(e_experiments)")
        }
    assert "source_state_id" in e_columns
    assert "output_description_json" in e_columns
    with sqlite3.connect(database.path) as connection:
        m_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(m_states)")
        }
    assert "world_prediction_json" in m_columns
    assert "goal_prediction_json" in m_columns
    assert "turn_metrics_json" in m_columns
    with sqlite3.connect(database.path) as connection:
        debug_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(model_input_debug_records)")
        }
    assert "m_state_id" in debug_columns
    assert "request_json" in debug_columns
    assert "usage_json" in debug_columns


def test_state_memory_prewrites_and_completes_source_rows(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state = StateMemory(database)
    observation = Observation(id="obs-current", step=0, frame={"frame": 0})
    action = ActionSpec(action_id="ACTION1")
    observation_ref = ObservationRef(memory="state", id=observation.id)
    trace = AgentTrace(
        step=0,
        first_observation_ref=observation_ref,
        current_observation_ref=observation_ref,
        final_action=action,
    )

    pending = state.prewrite_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        contexts=ContextDocuments(),
        metadata={"prewritten": True},
    )

    assert pending.chosen_action is None
    assert pending.agent_trace is None
    assert state.read_latest_state("game-1") is None
    assert state.list_states(game_id="game-1") == []
    assert state.read_state_source(pending.id).current_observation["id"] == (
        "obs-current"
    )

    completed = state.complete_state(
        state_id=pending.id,
        chosen_action=action,
        contexts=ContextDocuments(agent=RoleContext(game="updated")),
        agent_trace=trace,
        metadata={"prewritten": False},
    )

    assert completed.id == pending.id
    assert completed.chosen_action == {"action_id": "ACTION1", "data": None}
    assert completed.agent_trace is not None
    assert state.read_latest_state("game-1").id == pending.id
    assert state.list_states(game_id="game-1")[0].agent_context.game == "updated"


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
            predicted_description={"frame": "world"},
            source_observation_ref=observation_ref,
            action=action,
        ),
        goal_prediction=ToolResult(
            id="goal-post",
            tool="goal",
            predicted_description={"frame": "goal"},
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
        turn_metrics=TurnMetrics(
            time_cost=2.0,
            trace_cost=1.5,
            cumulative_score=3.0,
        ),
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
    assert latest.turn_metrics.time_cost == 2.0
    assert latest.turn_metrics.trace_cost == 1.5
    assert latest.turn_metrics.cumulative_score == 3.0
    assert first.world_prediction["id"] == "world-post"
    assert first.goal_prediction["id"] == "goal-post"
    assert first.turn_metrics == TurnMetrics()
    assert state.read_latest_state("never-played") is None

    first_debug = state.write_model_input_debug_record(
        m_state_id=first.id,
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        call_slot="agent",
        provider="openai",
        model="gpt-5-nano",
        phase="final_action",
        attempt=0,
        request={"model": "gpt-5-nano", "input": [{"role": "user"}]},
        usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        metadata={"response_id": "resp-1"},
    )
    second_debug = state.write_model_input_debug_record(
        m_state_id=second.id,
        run_id="run-2",
        game_id="game-1",
        turn_id=2,
        call_slot="world",
        provider="ollama",
        model="gemma4:e4b",
        phase="complete",
        attempt=0,
        request={"model": "gemma4:e4b", "messages": [{"role": "user"}]},
        usage={"prompt_eval_count": 5, "eval_count": 3},
    )

    assert first_debug.request["input"][0]["role"] == "user"
    assert state.list_model_input_debug_records(m_state_id=first.id)[0].usage == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert [
        record.id
        for record in state.list_model_input_debug_records(
            run_id="run-2",
            game_id="game-1",
            turn_id=2,
        )
    ] == [second_debug.id]

    state.cleanup_keep_latest_per_game()

    remaining = state.list_states()
    assert len(remaining) == 2
    assert first.id not in {record.id for record in remaining}
    assert state.list_model_input_debug_records(m_state_id=first.id) == []
    assert state.list_model_input_debug_records(m_state_id=second.id)[0].id == (
        second_debug.id
    )

    state.clear_states()
    assert state.list_states() == []
    assert state.list_model_input_debug_records() == []


def test_state_memory_reads_latest_general_contexts_across_games(tmp_path) -> None:
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

    assert state.read_latest_general_contexts() == ContextDocuments()

    state_record = state.write_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(
            world=RoleContext(general="world K 1", game="world L 1"),
            goal=RoleContext(general="goal K 1", game="goal L 1"),
            agent=RoleContext(general="agent K 1", game="agent L 1"),
        ),
        agent_trace=trace,
    )
    state.write_state(
        run_id="run-2",
        game_id="game-2",
        step=0,
        frame_index=0,
        frame_count=1,
        current_observation=observation,
        chosen_action=action,
        contexts=ContextDocuments(
            world=RoleContext(general="world K 2", game="world L 2"),
            goal=RoleContext(general="goal K 2", game="goal L 2"),
            agent=RoleContext(general="agent K 2", game="agent L 2"),
        ),
        agent_trace=trace,
    )

    contexts = state.read_latest_general_contexts()
    assert contexts.world == RoleContext(general="world K 2", game="")
    assert contexts.goal == RoleContext(general="goal K 2", game="")
    assert contexts.agent == RoleContext(general="agent K 2", game="")


def test_experimental_memory_writes_and_reads_description_experiments(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(
        tool="world",
        source_state_id=12,
        action=ActionSpec(action_id="ACTION1"),
    )
    result = ToolResult(
        id="world-result-1",
        tool="world",
        predicted_description={"frame": "predicted"},
        source_observation_ref=source_ref,
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
            frame=result.predicted_description,
            frames=(result.predicted_description,),
        ),
        tool_result=result,
    )

    latest = experimental.read_experiment(record.id)
    assert latest is not None
    assert latest.source_state_id == 12
    assert latest.tool_name == "world"
    assert latest.output_description["frame"] == {"frame": "predicted"}
    assert latest.tool_result["metadata"] == {"quality": "fake"}


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
        predicted_description=Image.new("RGB", (11, 13), color=(9, 8, 7)),
        source_observation_ref=source_ref,
        action=action,
    )
    experiment = experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="world", source_state_id=12, action=action),
        output_description=Observation(
            id=result.id,
            step=0,
            frame=result.predicted_description,
        ),
        tool_result=result,
    )
    stored = experimental.read_experiment(experiment.id)

    assert stored is not None
    assert isinstance(stored.output_description["frame"], Image.Image)
    assert stored.output_description["frame"].size == (64, 64)


def test_experimental_memory_cleanup_keeps_latest_turns_per_run_and_game(
    tmp_path,
) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    experimental = ExperimentalMemory(database)
    source_ref = ObservationRef(memory="state", id="obs-0")
    call = ToolCall(tool="goal", source_state_id=12)
    result = ToolResult(
        id="goal-result",
        tool="goal",
        predicted_description={"frame": "goal"},
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
                output_description=Observation(
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


def test_state_memory_cleanup_clears_auxiliary_memory_tables(tmp_path) -> None:
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
    state_record = state.write_state(
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
    state.write_model_input_debug_record(
        m_state_id=state_record.id,
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        call_slot="agent",
        provider="openai",
        model="gpt-5-nano",
        phase="final_action",
        attempt=0,
        request={"model": "gpt-5-nano"},
    )
    experimental.write_experiment(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        tool_call=ToolCall(tool="goal", source_state_id=12),
        output_description=Observation(id="goal-0", step=0, frame={"goal": True}),
        tool_result=ToolResult(
            id="goal-0",
            tool="goal",
            predicted_description={"goal": True},
            source_observation_ref=observation_ref,
        ),
    )

    state.clear_memory_tables()

    assert state.list_states() == []
    assert state.list_model_input_debug_records() == []
    assert experimental.list_experiments() == []
