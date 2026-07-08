"""Tests for SQLite-backed active state memory."""

from __future__ import annotations

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    Observation,
    ObservationRef,
    RoleContext,
    ToolCall,
    ToolResult,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    _agent_context_fields,
)


def test_m_state_schema_has_agent_context_only(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    database.initialize_schema()

    with sqlite3.connect(database.path) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(m_states)").fetchall()
        ]

    assert "agent_context_json" in columns
    assert "agent_trace_json" in columns
    assert "turn_metrics_json" in columns


def test_state_memory_prewrite_complete_and_list_agent_context(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(
        agent=RoleContext(general="agent K", game="agent L")
    )
    observation = _observation("obs-1")

    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
        contexts=contexts,
    )

    completed = memory.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=contexts,
        agent_trace=_trace(observation),
    )

    rows = memory.list_states(game_id="game-1")
    assert rows == [completed]
    assert rows[0].agent_context == contexts.agent
    assert rows[0].chosen_action["action_id"] == "ACTION1"
    assert rows[0].metadata["turn_id"] == 1


def test_hydrate_contexts_uses_latest_agent_general_and_game(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    defaults = ContextDocuments(agent=RoleContext(general="default K", game="default L"))

    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=ContextDocuments(
            agent=RoleContext(general="agent K", game="agent L")
        ),
        agent_trace=_trace(_observation("obs-1")),
    )

    hydrated = memory.hydrate_contexts_for_game(
        game_id="game-1",
        defaults=defaults,
    )

    assert hydrated.agent == RoleContext(general="agent K", game="agent L")


def test_update_state_contexts_updates_agent_context(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    row = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=ContextDocuments(agent=RoleContext(game="old")),
        agent_trace=_trace(_observation("obs-1")),
    )

    updated = memory.update_state_contexts(
        state_id=row.id,
        contexts=ContextDocuments(
            agent=RoleContext(general="new K", game="new L")
        ),
    )

    assert updated.agent_context == RoleContext(general="new K", game="new L")


def test_agent_context_fields_accepts_single_probing_field_for_merge() -> None:
    assert _agent_context_fields(
        '{"probing_strategy": "probe left corridor"}',
        expected_keys=("probing_strategy",),
    ) == {"probing_strategy": "probe left corridor"}


def test_agent_context_fields_accepts_single_policy_field_for_merge() -> None:
    assert _agent_context_fields(
        '{"policy_strategy": "push toward exit"}',
        expected_keys=("policy_strategy",),
    ) == {"policy_strategy": "push toward exit"}


def test_read_agent_context_history_uses_updater_summary_snapshots(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="agent L"))

    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-1")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "try each action",
                "policy_strategy": "",
            }
        },
    )
    memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-2"),
        chosen_action=ActionSpec(action_id="ACTION2"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-2")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "try each action",
                "policy_strategy": "move toward exit",
            }
        },
    )
    current = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=3,
        current_observation=_observation("obs-3"),
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
        contexts=contexts,
    )

    history = memory.read_agent_context_history(
        game_id="game-1",
        run_id="run-1",
        before_state_id=current.id,
        limit=2,
    )

    assert history == (
        '{\n  "probing_strategy": "try each action",\n'
        '  "policy_strategy": ""\n}',
        '{\n  "probing_strategy": "try each action",\n'
        '  "policy_strategy": "move toward exit"\n}',
    )


def test_level_solution_summary_and_strategy_interval_round_trip(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="agent L"))
    first = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-1")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "probe the switch",
                "policy_strategy": "",
            }
        },
    )
    second = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-2"),
        chosen_action=ActionSpec(action_id="ACTION2"),
        contexts=contexts,
        agent_trace=_trace(_observation("obs-2")),
        metadata={
            "agent_context_history": {
                "probing_strategy": "probe the switch",
                "policy_strategy": "walk to the goal",
            }
        },
    )

    summary = memory.write_level_solution_summary(
        run_id="run-1",
        game_id="game-1",
        completed_level=1,
        source_state_ids=(first.id, second.id),
        solution_method="Probe the switch, then walk to the goal.",
        metadata={"source": "test"},
    )
    history = memory.read_agent_strategy_history_between(
        game_id="game-1",
        run_id="run-1",
        after_state_id=first.id,
        through_state_id=second.id,
    )

    assert memory.read_latest_level_solution_summary(
        run_id="run-1",
        game_id="game-1",
    ) == summary
    assert history == (
        '{\n  "probing_strategy": "probe the switch",\n'
        '  "policy_strategy": "walk to the goal"\n}',
    )


def test_e_experiment_persists_generic_tool_result(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    database.initialize_schema()
    result = ToolResult(
        id="tool-result-1",
        tool="inspect",
        output={"value": 7},
        source_observation_ref=ObservationRef(memory="state", id="obs-1"),
    )

    stored = database.write_e_experiment(
        game_id="game-1",
        run_id="run-1",
        turn_id=1,
        tool_name="inspect",
        source_state_id=12,
        tool_call=ToolCall(tool="inspect", source_state_id=12),
        output_description=result.output,
        tool_result=result,
    )

    assert stored.tool_name == "inspect"
    assert stored.output_description == {"value": 7}
    assert stored.tool_result["output"] == {"value": 7}


def test_model_input_debug_records_round_trip(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    row = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        contexts=ContextDocuments(agent=RoleContext(game="agent L")),
        agent_trace=_trace(_observation("obs-1")),
    )

    record = memory.write_model_input_debug_record(
        m_state_id=row.id,
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        call_slot="agent",
        provider="fake",
        model="fake-model",
        phase="primary",
        attempt=0,
        request={"messages": []},
        metadata={"active": True},
    )

    records = memory.database.list_model_input_debug_records(m_state_id=row.id)
    assert records == [record]
    assert records[0].metadata == {"active": True}


def _observation(observation_id: str) -> Observation:
    return Observation(
        id=observation_id,
        step=1,
        frame=Image.new("RGB", (8, 8), color=(1, 2, 3)),
    )


def _trace(observation: Observation) -> AgentTrace:
    ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=ActionSpec(action_id="ACTION1"),
    )
