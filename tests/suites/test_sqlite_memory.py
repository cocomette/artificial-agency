"""Tests for SQLite-backed active state memory."""

from __future__ import annotations

import sqlite3

from PIL import Image

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    GoalPrediction,
    Observation,
    ObservationRef,
    RewardJudgeScore,
    RoleContext,
    ToolCall,
    ToolResult,
    TurnReward,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory


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


def test_v1_runtime_tables_round_trip(tmp_path) -> None:
    memory = StateMemory(SQLiteDatabase(tmp_path / "memory.sqlite"))
    action = ActionSpec(action_id="ACTION1")
    goal = GoalPrediction(
        goal="solve",
        subgoals=("open path",),
        steps_remaining=3,
        confidence=0.75,
    )
    reward = TurnReward(
        prediction_accuracy=0.9,
        learning_progress=None,
        goal_delta=0.25,
        progress_bonus=0.0,
        resource_cost=0.0,
        lp_weight=0.8,
        goal_weight=0.2,
        total=0.77,
    )
    judge = RewardJudgeScore(score=0.9, notes="close", error_tags=("minor",))

    candidate = memory.write_candidate_prediction(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        candidate_index=0,
        action=action,
        prediction="object moved",
        source="runtime_simple_action",
    )
    score = memory.write_judge_score(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        candidate_prediction_id=candidate.id,
        score=judge.score,
        notes=judge.notes,
        error_tags=judge.error_tags,
    )
    goal_record = memory.write_goal_prediction(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        goal_prediction=goal,
        memory_document="memory",
    )
    reward_record = memory.write_reward(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        reward=reward,
    )
    ledger = memory.write_turn_ledger(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        m_state_id=None,
        action=action,
        change_summary="object moved",
        memory_document="memory",
        goal_prediction=goal,
        reward=reward,
    )
    replay = memory.write_replay_sample(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        role="world",
        prompt={"prompt": "predict"},
        completion={"target": "object moved"},
        reward=0.9,
        held_out=False,
        metadata={"base_model": "qwen"},
    )
    update = memory.write_lora_update(
        run_id="run-1",
        game_id="game-1",
        update_index=1,
        role="world",
        status="queued",
        adapter_name="world-lora",
        adapter_path="/tmp/world-lora",
    )

    assert candidate.prediction == "object moved"
    assert score.candidate_prediction_id == candidate.id
    assert score.error_tags == ("minor",)
    assert goal_record.goal_prediction["goal"] == "solve"
    assert reward_record.reward["total"] == 0.77
    assert ledger.change_summary == "object moved"
    assert memory.list_replay_samples(
        run_id="run-1",
        game_id="game-1",
        role="world",
    ) == [replay]
    assert memory.list_replay_samples(
        run_id="run-1",
        game_id="game-1",
        role="world",
        held_out=False,
        after_id=0,
        ascending=True,
    ) == [replay]
    assert memory.list_replay_samples(
        run_id="run-1",
        game_id="game-1",
        role="world",
        held_out=True,
    ) == []
    assert memory.list_replay_samples(
        run_id="run-1",
        game_id="game-1",
        role="world",
        after_id=replay.id,
    ) == []
    updated_replay = memory.update_replay_sample_reward_metadata(
        sample_id=replay.id,
        reward=-0.25,
        metadata={**replay.metadata, "learning_progress_backfill": {"lp": -0.25}},
    )
    assert updated_replay.reward == -0.25
    assert updated_replay.metadata["learning_progress_backfill"] == {"lp": -0.25}
    assert update.adapter_name == "world-lora"


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
