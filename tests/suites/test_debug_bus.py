"""Tests for the production-facing debug bus."""

from __future__ import annotations

from io import StringIO

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    DebugEvent,
    FrameDecisionRecorded,
    FrameTurnCompleted,
    ModelCallCompleted,
)
from face_of_agi.debug.sinks import DebugTrace, LiveTurnMonitor
from face_of_agi.memory import SQLiteDatabase, StateMemory


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[DebugEvent] = []

    def emit(self, event: DebugEvent) -> None:
        self.events.append(event)


class CapturingAdapter:
    def __init__(self) -> None:
        self._model_input_debug_records = [
            {
                "call_slot": "agent",
                "provider": "openai",
                "model": "gpt-5-nano",
                "phase": "final_action",
                "attempt": 0,
                "request": {"input": [{"role": "user"}]},
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "metadata": {"response_id": "resp-1"},
            }
        ]


def test_debug_bus_emits_typed_events_to_sink() -> None:
    sink = CollectingSink()
    event = FrameDecisionRecorded(
        frame_turn=1,
        frame_context=_frame_context(source_state_id=7),
        action=ActionSpec(action_id="ACTION1"),
        trace=_agent_trace(),
    )

    DebugBus(sink=sink).emit(event)

    assert sink.events == [event]


def test_debug_bus_persists_existing_model_input_records(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state_memory = StateMemory(database)
    source = state_memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        frame_index=0,
        frame_count=1,
        current_observation=Observation(id="obs-0", step=0, frame={"frame": 0}),
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
        learner_snapshot={"buffer": {"size": 0}},
    )
    adapter = CapturingAdapter()

    DebugBus(state_memory=state_memory).capture_model_inputs(
        _frame_context(source_state_id=source.id),
        1,
        adapter,
    )

    records = state_memory.list_model_input_debug_records(m_state_id=source.id)
    assert len(records) == 1
    assert records[0].request["input"][0]["role"] == "user"
    assert adapter._model_input_debug_records == []


def test_debug_bus_can_drain_model_input_records_without_persisting(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "memory.sqlite")
    state_memory = StateMemory(database)
    source = state_memory.prewrite_frame_turn_source(
        run_id="run-1",
        game_id="game-1",
        turn_id=1,
        frame_index=0,
        frame_count=1,
        current_observation=Observation(id="obs-0", step=0, frame={"frame": 0}),
        control_mode=FrameControlMode.real_environment_turn(
            (ActionSpec(action_id="ACTION1"),)
        ),
        learner_snapshot={"buffer": {"size": 0}},
    )
    adapter = CapturingAdapter()

    DebugBus(
        state_memory=state_memory,
        persist_model_input_debug_records=False,
    ).capture_model_inputs(
        _frame_context(source_state_id=source.id),
        1,
        adapter,
    )

    assert state_memory.list_model_input_debug_records(m_state_id=source.id) == []
    assert adapter._model_input_debug_records == []


def test_debug_trace_renders_frame_decision_event() -> None:
    output = StringIO()
    trace = DebugTrace(mode="minimal", color="never", output=output)

    trace.emit(
        FrameDecisionRecorded(
            frame_turn=1,
            frame_context=_frame_context(source_state_id=7),
            action=ActionSpec(action_id="ACTION1"),
            trace=_agent_trace(),
        )
    )

    rendered = output.getvalue()
    assert "frame turn 1" in rendered
    assert "action: online learner selected ACTION1" in rendered


def test_debug_trace_ignores_model_call_timing_event() -> None:
    output = StringIO()
    trace = DebugTrace(mode="minimal", color="never", output=output)

    trace.emit(ModelCallCompleted(role="agent", duration_seconds=1.0))

    assert output.getvalue() == ""


def test_live_turn_monitor_prints_aggregate_after_selected_game_cadence() -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=3, output=output)

    monitor.emit(ModelCallCompleted(role="backbone", duration_seconds=0.5))
    monitor.emit(ModelCallCompleted(role="planner", duration_seconds=1.0))
    monitor.emit(ModelCallCompleted(role="replay", duration_seconds=1.5))
    monitor.emit(_completed_turn(game_index=1, duration=1.0, completed_levels=1))
    monitor.emit(
        _completed_turn(
            game_index=1,
            duration=2.0,
            completed_levels=2,
            controllable=False,
        )
    )

    assert output.getvalue() == ""

    monitor.emit(_completed_turn(game_index=2, duration=3.0, completed_levels=3))

    rendered = output.getvalue()
    assert rendered.startswith("throughput:")
    assert "turns=3" in rendered
    assert "games=3" in rendered
    assert "avg_turn_sec=2.000" in rendered
    assert "min_turn_sec=1.000" in rendered
    assert "max_turn_sec=3.000" in rendered
    assert "avg_turn_sec_backbone=0.500" in rendered
    assert "avg_turn_sec_planner=1.000" in rendered
    assert "avg_turn_sec_replay=1.500" in rendered
    assert "avg_controllable_turns_per_game=0.67" in rendered
    assert "total_completed_levels=5" in rendered


def _frame_context(*, source_state_id: int | None) -> FrameTurnContext:
    observation = Observation(id="obs-0", step=0, frame={"frame": 0})
    ref = ObservationRef(memory="state", id=observation.id)
    action = ActionSpec(action_id="ACTION1")
    return FrameTurnContext(
        run_id="run-1",
        game_id="game-1",
        first_observation_ref=ref,
        current_observation_ref=ref,
        current_observation=observation,
        current_source_state_id=source_state_id,
        frame_index=0,
        frame_count=1,
        control_mode=FrameControlMode.real_environment_turn((action,)),
    )


def _agent_trace() -> AgentTrace:
    action = ActionSpec(action_id="ACTION1")
    ref = ObservationRef(memory="state", id="obs-0")
    return AgentTrace(
        step=0,
        first_observation_ref=ref,
        current_observation_ref=ref,
        final_action=action,
        reasoning_summary="test trace",
    )


def _completed_turn(
    *,
    game_index: int,
    duration: float,
    completed_levels: int,
    controllable: bool = True,
) -> FrameTurnCompleted:
    return FrameTurnCompleted(
        run_id=f"run-{game_index}",
        game_id=f"game-{game_index}",
        game_index=game_index,
        turn_id=1,
        env_step=0,
        frame_index=0,
        frame_count=1,
        controllable=controllable,
        action=ActionSpec(action_id="ACTION1"),
        turn_duration_seconds=duration,
        completed_levels=completed_levels,
        remaining_actions=0,
    )
