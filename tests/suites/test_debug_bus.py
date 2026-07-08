"""Tests for the production-facing debug bus."""

from __future__ import annotations

from io import StringIO

from PIL import Image

import face_of_agi.debug.sinks.live_turn_monitor as live_turn_monitor_module
from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    FrameControlMode,
    FrameTurnContext,
    Observation,
    ObservationRef,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    DebugEvent,
    EnvironmentStepRecorded,
    FrameDecisionRecorded,
    FrameTurnCompleted,
    KnownStateSimulationCompleted,
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
    source = state_memory.prewrite_state(
        run_id="run-1",
        game_id="game-1",
        step=0,
        frame_index=0,
        frame_count=1,
            current_observation=Observation(
                id="obs-0",
                step=0,
                frame=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            ),
        contexts=ContextDocuments(),
        metadata={"turn_id": 1},
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
    assert "action: X selected ACTION1" in rendered


def test_debug_trace_ignores_model_call_timing_event() -> None:
    output = StringIO()
    trace = DebugTrace(mode="minimal", color="never", output=output)

    trace.emit(ModelCallCompleted(role="world", duration_seconds=1.0))

    assert output.getvalue() == ""


def test_live_turn_monitor_prints_aggregate_after_selected_game_cadence() -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=5, output=output)

    monitor.emit(_completed_turn(game_index=1, duration=1.0, completed_levels=0))
    monitor.emit(
        _completed_turn(
            game_index=1,
            duration=2.0,
            completed_levels=0,
            controllable=False,
        )
    )

    assert output.getvalue() == ""

    monitor.emit(ModelCallCompleted(role="change", duration_seconds=4.0))
    monitor.emit(ModelCallCompleted(role="change", duration_seconds=6.0))
    monitor.emit(ModelCallCompleted(role="compacter", duration_seconds=8.0))
    monitor.emit(ModelCallCompleted(role="compacter", duration_seconds=10.0))
    monitor.emit(ModelCallCompleted(role="agent", duration_seconds=12.0))
    monitor.emit(
        EnvironmentStepRecorded(
            action=ActionSpec(action_id="ACTION1"),
            next_observation=Observation(id="next-1", step=1, frame={}),
            remaining_actions=9,
        )
    )
    monitor.emit(
        EnvironmentStepRecorded(
            action=ActionSpec(action_id="ACTION1"),
            next_observation=Observation(id="next-2", step=2, frame={}),
            remaining_actions=8,
        )
    )
    monitor.emit(
        KnownStateSimulationCompleted(
            run_id="run-2",
            game_id="game-2",
            game_index=2,
            turn_id=3,
            duration_seconds=5.0,
            simulated_row_count=2,
            simulated_action_count=1,
            catchup_action_count=1,
            saved_environment_action_count=0,
        )
    )
    monitor.emit(_completed_turn(game_index=2, duration=3.0, completed_levels=0))

    rendered = output.getvalue()
    assert rendered.startswith("throughput:")
    assert "turns=5" in rendered
    assert "games=5" in rendered
    assert "avg_turn_sec=2.200" in rendered
    assert "min_turn_sec=" not in rendered
    assert "max_turn_sec=" not in rendered
    assert "avg_turn_sec_change=5.000" in rendered
    assert "avg_turn_sec_compacter=9.000" in rendered
    assert "avg_turn_sec_agent=12.000" in rendered
    assert "avg_controllable_actions_per_game=0.80" in rendered
    assert "avg_controllable_submitted_actions_per_game=0.40" in rendered
    assert "total_completed_levels=0" in rendered


def test_live_turn_monitor_turn_count_and_rate_count_simulated_rows(
    monkeypatch,
) -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=4, output=output)
    monotonic_times = iter((101.0, 130.0))
    monkeypatch.setattr(
        live_turn_monitor_module.time,
        "monotonic",
        lambda: next(monotonic_times),
    )

    monitor.emit(_completed_turn(game_index=1, duration=1.0, completed_levels=0))
    monitor.emit(
        KnownStateSimulationCompleted(
            run_id="run-1",
            game_id="game-1",
            game_index=1,
            turn_id=2,
            duration_seconds=9.0,
            simulated_row_count=3,
            simulated_action_count=2,
            catchup_action_count=1,
            saved_environment_action_count=1,
        )
    )

    rendered = output.getvalue()
    assert "turns=4" in rendered
    assert "avg_turn_sec=2.500" in rendered
    assert "turns_per_min=8.00" in rendered


def test_live_turn_monitor_prints_level_completion_immediately() -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=3, output=output)

    monitor.emit(
        _completed_turn(
            game_index=1,
            duration=1.0,
            completed_levels=1,
            remaining_actions=7,
            max_actions_per_level=10,
        )
    )

    rendered = output.getvalue()
    assert rendered.startswith("level_completed:")
    assert "game_id=game-1" in rendered
    assert "game_index=1" in rendered
    assert "run_id=run-1" in rendered
    assert "completed_levels=1" in rendered
    assert "completed_level_delta=1" in rendered
    assert "controllable_actions_for_level=3" in rendered
    assert "throughput:" not in rendered

    monitor.emit(
        _completed_turn(
            game_index=1,
            duration=1.0,
            completed_levels=1,
            remaining_actions=6,
            max_actions_per_level=10,
        )
    )

    assert output.getvalue().count("level_completed:") == 1


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
    remaining_actions: int = 0,
    max_actions_per_level: int | None = 1,
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
        remaining_actions=remaining_actions,
        max_actions_per_level=max_actions_per_level,
    )
