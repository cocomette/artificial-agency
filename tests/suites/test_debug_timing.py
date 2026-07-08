"""Tests for debug timing telemetry."""

from __future__ import annotations

from io import StringIO

from face_of_agi.contracts import ActionSpec
from face_of_agi.debug.events import (
    FrameTurnCompleted,
    ModelCallCompleted,
    ModelCallEventRecorded,
)
from face_of_agi.debug.sinks.live_turn_monitor import LiveTurnMonitor


def _completed_turn(
    *,
    turn_id: int,
    controllable: bool,
    duration: float,
    completed_levels: int,
) -> FrameTurnCompleted:
    return FrameTurnCompleted(
        run_id="run-1",
        game_id=f"game-{turn_id}",
        game_index=turn_id,
        turn_id=turn_id,
        env_step=turn_id,
        frame_index=0,
        frame_count=1,
        controllable=controllable,
        action=ActionSpec("ACTION1"),
        turn_duration_seconds=duration,
        completed_levels=completed_levels,
        remaining_actions=10,
    )


def test_live_turn_monitor_reports_model_call_averages() -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=2, output=output)

    monitor.emit(ModelCallCompleted(role="agent", duration_seconds=2.0))
    monitor.emit(ModelCallCompleted(role="agent", duration_seconds=4.0))
    monitor.emit(ModelCallCompleted(role="change", duration_seconds=8.0))
    monitor.emit(ModelCallCompleted(role="ignored", duration_seconds=100.0))
    monitor.emit(
        _completed_turn(
            turn_id=1,
            controllable=True,
            duration=10.0,
            completed_levels=1,
        )
    )
    assert output.getvalue() == ""

    monitor.emit(
        _completed_turn(
            turn_id=2,
            controllable=False,
            duration=20.0,
            completed_levels=2,
        )
    )

    line = output.getvalue()
    assert "turns=2" in line
    assert "avg_turn_sec=15.000" in line
    assert "avg_model_sec_agent=3.000" in line
    assert "avg_model_sec_change=8.000" in line
    assert "avg_model_sec_historizer=0.000" in line
    assert "avg_model_sec_memory=0.000" in line
    assert "avg_controllable_turns_per_game=0.50" in line
    assert "total_completed_levels=3" in line


def test_live_turn_monitor_reports_model_repair_attempts() -> None:
    output = StringIO()
    monitor = LiveTurnMonitor(selected_game_count=1, output=output)

    monitor.emit(
        ModelCallEventRecorded(
            role="agent",
            provider="vllm",
            model="qwen",
            event="repair_attempt",
            status="started",
            game_id="abc123",
            turn_id=7,
            metadata={
                "attempt": 2,
                "validation_error_type": "AgentOutputError",
            },
        )
    )
    monitor.emit(
        ModelCallEventRecorded(
            role="change",
            provider="vllm",
            model="qwen",
            event="provider_end",
            status="success",
            game_id="abc123",
            turn_id=7,
        )
    )

    lines = output.getvalue().splitlines()
    assert lines == [
        "model_repair_attempt: role=agent game_id=abc123 turn=7 provider=vllm "
        "model=qwen attempt=2 error_type=AgentOutputError"
    ]
