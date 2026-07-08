"""Live aggregate throughput monitor for runtime frame turns."""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from face_of_agi.debug.events import DebugEvent, FrameTurnCompleted, ModelCallCompleted

_MODEL_AVG_FIELDS = {
    "change": "avg_turn_sec_change",
    "historizer": "avg_turn_sec_hist",
    "world": "avg_turn_sec_world",
    "agent_probing": "avg_turn_sec_probing",
    "agent_policy": "avg_turn_sec_policy",
}


class LiveTurnMonitor:
    """Print aggregate frame-turn throughput at a fixed completed-turn cadence."""

    def __init__(
        self,
        *,
        selected_game_count: int = 1,
        output: TextIO | None = None,
    ) -> None:
        if selected_game_count < 1:
            raise ValueError("selected_game_count must be at least 1")
        self.selected_game_count = selected_game_count
        self.output = output or sys.stdout
        self._lock = threading.Lock()
        self._turn_count = 0
        self._controllable_turn_count = 0
        self._total_turn_seconds = 0.0
        self._model_call_totals = {role: 0.0 for role in _MODEL_AVG_FIELDS}
        self._model_call_counts = {role: 0 for role in _MODEL_AVG_FIELDS}
        self._started_at: float | None = None
        self._completed_levels_by_game: dict[tuple[int | None, str], int] = {}

    def emit(self, event: DebugEvent) -> None:
        """Record completed frame turns and print cadence summaries."""

        if isinstance(event, ModelCallCompleted):
            self._record_model_call(event)
            return
        if not isinstance(event, FrameTurnCompleted):
            return

        with self._lock:
            now = time.monotonic()
            output_lines: list[str] = []
            duration = max(0.0, float(event.turn_duration_seconds))
            if self._started_at is None:
                self._started_at = now - duration

            self._turn_count += 1
            if event.controllable:
                self._controllable_turn_count += 1
            self._total_turn_seconds += duration
            game_key = (event.game_index, event.game_id)
            previous_completed_levels = self._completed_levels_by_game.get(
                game_key,
                0,
            )
            self._completed_levels_by_game[game_key] = event.completed_levels
            if event.completed_levels > previous_completed_levels:
                output_lines.append(
                    self._level_completed_line(
                        event,
                        previous_completed_levels=previous_completed_levels,
                    )
                )

            if self._turn_count % self.selected_game_count == 0:
                output_lines.append(self._summary_line(now))
            if output_lines:
                self.output.write("\n".join(output_lines) + "\n")
                self.output.flush()

    def _summary_line(self, now: float) -> str:
        avg_turn_seconds = self._total_turn_seconds / self._turn_count
        elapsed_seconds = (
            max(0.0, now - self._started_at)
            if self._started_at is not None
            else 0.0
        )
        turns_per_minute = (
            (self._turn_count / elapsed_seconds) * 60.0
            if elapsed_seconds > 0
            else 0.0
        )
        avg_controllable_turns_per_game = (
            self._controllable_turn_count / self.selected_game_count
        )
        total_completed_levels = sum(self._completed_levels_by_game.values())
        return (
            "throughput:"
            f" turns={self._turn_count}"
            f" games={self.selected_game_count}"
            f" avg_turn_sec={avg_turn_seconds:.3f}"
            f"{self._model_averages_text()}"
            f" turns_per_min={turns_per_minute:.2f}"
            f" avg_controllable_turns_per_game={avg_controllable_turns_per_game:.2f}"
            f" total_completed_levels={total_completed_levels}"
        )

    def _level_completed_line(
        self,
        event: FrameTurnCompleted,
        *,
        previous_completed_levels: int,
    ) -> str:
        actions_for_level = _controllable_actions_for_level(event)
        return (
            "level_completed:"
            f" game_id={event.game_id}"
            f" game_index={_value_or_none(event.game_index)}"
            f" run_id={event.run_id}"
            f" completed_levels={event.completed_levels}"
            f" completed_level_delta={event.completed_levels - previous_completed_levels}"
            f" controllable_actions_for_level={actions_for_level}"
        )

    def _record_model_call(self, event: ModelCallCompleted) -> None:
        if event.role not in _MODEL_AVG_FIELDS:
            return
        with self._lock:
            duration = max(0.0, float(event.duration_seconds))
            self._model_call_totals[event.role] += duration
            self._model_call_counts[event.role] += 1

    def _model_averages_text(self) -> str:
        fields = []
        for role, field in _MODEL_AVG_FIELDS.items():
            count = self._model_call_counts[role]
            average = self._model_call_totals[role] / count if count else 0.0
            fields.append(f" {field}={average:.3f}")
        return "".join(fields)


def _controllable_actions_for_level(event: FrameTurnCompleted) -> str:
    if event.max_actions_per_level is None:
        return "unknown"
    used = int(event.max_actions_per_level) - int(event.remaining_actions)
    return str(max(0, used))


def _value_or_none(value: object | None) -> str:
    return "none" if value is None else str(value)
