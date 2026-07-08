"""Tests for SQLite-backed active state memory."""

from __future__ import annotations

import sqlite3

from arcengine import GameAction
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
            (ActionSpec(action_id=GameAction.ACTION1),)
        ),
        contexts=contexts,
    )

    completed = memory.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id=GameAction.ACTION1),)
        ),
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=ActionSpec(action_id=GameAction.ACTION1),
        contexts=contexts,
        agent_trace=_trace(observation),
    )

    rows = memory.list_states(game_id="game-1")
    assert rows == [completed]
    assert rows[0].agent_context == contexts.agent
    assert rows[0].chosen_action["action_id"] == "ACTION1"
    assert (
        rows[0].metadata["control_mode"]["allowed_actions"][0]["action_id"]
        == "ACTION1"
    )
    assert rows[0].metadata["turn_id"] == 1


def test_state_memory_merges_metadata_into_existing_row(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="agent L"))
    observation = _observation("obs-merge")
    action = ActionSpec(action_id=GameAction.ACTION1)
    control_mode = FrameControlMode.real_environment_turn((action,))

    source = memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        current_observation=observation,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        contexts=contexts,
    )
    completed = memory.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=control_mode,
        previous_observation_ref=None,
        recent_action_history=(),
        chosen_action=action,
        contexts=contexts,
        agent_trace=_trace(observation),
        extra_metadata={"existing": {"a": 1}},
    )

    updated = memory.merge_state_metadata(
        state_id=completed.id,
        metadata={"known_state_simulation_catchup": {"successful": True}},
    )

    assert updated.metadata["turn_id"] == 1
    assert updated.metadata["existing"] == {"a": 1}
    assert updated.metadata["known_state_simulation_catchup"] == {
        "successful": True
    }


def test_state_memory_persists_action6_target_value_without_bbox(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    contexts = ContextDocuments(agent=RoleContext(game="agent L"))
    observation = _observation("obs-action6")
    action = ActionSpec(
        action_id="ACTION6",
        data={"x": 12, "y": 34},
        target="blue object",
        target_value=3,
        target_bbox=(100, 200, 300, 400),
    )
    history_entry = ActionHistoryEntry(
        action=action,
        controllable=True,
        changed_pixel_count=1,
        change_summary="Clicked blue object.",
    )

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
    completed = memory.complete_frame_turn_state(
        state_id=source.id,
        turn_id=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
        previous_observation_ref=None,
        recent_action_history=(history_entry,),
        chosen_action=action,
        contexts=contexts,
        agent_trace=AgentTrace(
            step=observation.step,
            first_observation_ref=ObservationRef(memory="state", id=observation.id),
            current_observation_ref=ObservationRef(memory="state", id=observation.id),
            final_action=action,
        ),
    )

    assert completed.chosen_action == {
        "action_id": "ACTION6",
        "data": {"x": 12, "y": 34},
        "target": "blue object",
        "target_value": 3,
    }
    assert completed.agent_trace is not None
    assert completed.agent_trace["final_action"]["target_value"] == 3
    assert "target_bbox" not in completed.agent_trace["final_action"]
    stored_history_action = completed.metadata["recent_action_history"][0]["action"]
    assert stored_history_action["target_value"] == 3
    assert "target_bbox" not in stored_history_action


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


def test_agent_context_fields_accepts_current_strategy_field() -> None:
    assert _agent_context_fields(
        '{"current_strategy": "push toward exit"}',
        expected_keys=("current_strategy",),
    ) == {"current_strategy": "push toward exit"}


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
                "current_strategy": "",
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
                "current_strategy": "move toward exit",
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
        '{\n  "current_strategy": ""\n}',
        '{\n  "current_strategy": "move toward exit"\n}',
    )


def test_compacter_level_summary_and_strategy_interval_round_trip(tmp_path) -> None:
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
                "current_strategy": "",
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
                "current_strategy": "walk to the goal",
            }
        },
    )

    summary = memory.write_compacter_level_summary(
        run_id="run-1",
        game_id="game-1",
        completed_level=1,
        source_state_ids=(first.id, second.id),
        previous_actions_summary="Probe the switch, then walk to the goal.",
        previous_strategy_summary="Switch strategy solved the level.",
        metadata={"source": "test"},
    )
    history = memory.read_agent_strategy_history_between(
        game_id="game-1",
        run_id="run-1",
        after_state_id=first.id,
        through_state_id=second.id,
    )

    assert memory.read_latest_compacter_level_summary(
        run_id="run-1",
        game_id="game-1",
    ) == summary
    assert summary.previous_actions_summary == "Probe the switch, then walk to the goal."
    assert summary.previous_strategy_summary == "Switch strategy solved the level."
    assert history == (
        '{\n  "current_strategy": "walk to the goal"\n}',
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


def _border_observation(
    observation_id: str,
    *,
    border_color: tuple[int, int, int],
) -> Observation:
    frame = Image.new("RGB", (64, 64), color=(1, 2, 3))
    for index in range(64):
        frame.putpixel((index, 0), border_color)
        frame.putpixel((index, 1), border_color)
        frame.putpixel((index, 2), border_color)
        frame.putpixel((index, 3), border_color)
        frame.putpixel((index, 60), border_color)
        frame.putpixel((index, 61), border_color)
        frame.putpixel((index, 62), border_color)
        frame.putpixel((index, 63), border_color)
        frame.putpixel((0, index), border_color)
        frame.putpixel((1, index), border_color)
        frame.putpixel((2, index), border_color)
        frame.putpixel((3, index), border_color)
        frame.putpixel((60, index), border_color)
        frame.putpixel((61, index), border_color)
        frame.putpixel((62, index), border_color)
        frame.putpixel((63, index), border_color)
    return Observation(id=observation_id, step=1, frame=frame)


def _trace(observation: Observation) -> AgentTrace:
    ref = ObservationRef(memory="state", id=observation.id)
    return AgentTrace(
        step=observation.step,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=ActionSpec(action_id="ACTION1"),
    )
