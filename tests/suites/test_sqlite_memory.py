"""Tests for SQLite-backed online learner state memory."""

from __future__ import annotations

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameControlMode,
    LearnerTurnTrace,
    Observation,
    ObservationRef,
    PlannerCandidate,
    ReplayStats,
    TransitionRecord,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory


def test_m_state_schema_has_learner_payload_columns(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    database.initialize_schema()

    with sqlite3.connect(database.path) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(m_states)").fetchall()
        ]

    assert "learner_snapshot_json" in columns
    assert "learner_trace_json" in columns
    assert "turn_metrics_json" in columns


def test_state_memory_prewrite_complete_and_list_learner_trace(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    observation = _observation("obs-1")
    trace = _learner_trace(observation)

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
        learner_snapshot={"buffer": {"size": 0}},
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
        learner_snapshot={"buffer": {"size": 1}},
        learner_trace=trace,
    )

    rows = memory.list_states(game_id="game-1")
    assert rows == [completed]
    assert rows[0].learner_snapshot["buffer"]["size"] == 1
    assert rows[0].learner_trace["transition"]["prediction_error"] == 0.25
    assert rows[0].learner_trace["planner_candidates"][0]["action"]["action_id"] == "ACTION1"
    assert rows[0].chosen_action["action_id"] == "ACTION1"
    assert rows[0].metadata["turn_id"] == 1


def test_read_latest_state_uses_newest_complete_row(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    first = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=1,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-1"),
        chosen_action=ActionSpec(action_id="ACTION1"),
        learner_snapshot={"frame_turn_count": 1},
        learner_trace=_learner_trace(_observation("obs-1")),
    )
    second = memory.write_state(
        run_id="run-1",
        game_id="game-1",
        step=2,
        frame_index=0,
        frame_count=1,
        current_observation=_observation("obs-2"),
        chosen_action=ActionSpec(action_id="ACTION2"),
        learner_snapshot={"frame_turn_count": 2},
        learner_trace=_learner_trace(_observation("obs-2"), action_id="ACTION2"),
    )

    assert first.id != second.id
    latest = memory.read_latest_state("game-1")

    assert latest is not None
    assert latest.id == second.id
    assert latest.learner_snapshot["frame_turn_count"] == 2


def test_learner_artifact_persists_generic_payload(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))

    stored = memory.write_learner_artifact(
        game_id="game-1",
        run_id="run-1",
        turn_id=1,
        kind="planner_debug",
        payload={"candidate_count": 3},
        metadata={"source": "unit"},
    )

    assert stored["kind"] == "planner_debug"
    assert stored["payload"] == {"candidate_count": 3}
    assert memory.list_learner_artifacts(run_id="run-1") == [stored]


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
        learner_snapshot={"buffer": {"size": 1}},
        learner_trace=_learner_trace(_observation("obs-1")),
    )

    record = memory.write_model_input_debug_record(
        m_state_id=row.id,
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        call_slot="backbone",
        provider="transformers",
        model="local-model",
        phase="encode",
        attempt=0,
        request={"observation_id": "obs-1"},
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


def _learner_trace(
    observation: Observation,
    *,
    action_id: str = "ACTION1",
) -> LearnerTurnTrace:
    action = ActionSpec(action_id=action_id)
    ref = ObservationRef(memory="state", id=observation.id)
    decision = DecisionResult(
        final_action=action,
        trace=AgentTrace(
            step=observation.step,
            first_observation_ref=ref,
            current_observation_ref=ref,
            final_action=action,
        ),
    )
    return LearnerTurnTrace(
        decision=decision,
        transition=TransitionRecord(
            previous_observation_ref=ref,
            next_observation_ref=ObservationRef(memory="state", id=f"{observation.id}-next"),
            action=action,
            controllable=True,
            changed_pixel_percent=12.5,
            prediction_error=0.25,
        ),
        replay=ReplayStats(real_update_count=1, replay_update_count=2),
        planner_candidates=(
            PlannerCandidate(action=action, score=1.0, predicted_value=0.5),
        ),
        backbone_metadata={"previous": {"backend": "fake"}},
    )
