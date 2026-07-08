"""Tests for the production-facing debug bus."""

from __future__ import annotations

from io import StringIO

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
from face_of_agi.debug.events import DebugEvent, FrameDecisionRecorded
from face_of_agi.debug.sinks import DebugTrace
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
        current_observation=Observation(id="obs-0", step=0, frame={"frame": 0}),
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
