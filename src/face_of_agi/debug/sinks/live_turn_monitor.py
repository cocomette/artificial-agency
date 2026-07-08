"""Live aggregate throughput monitor for runtime frame turns."""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from face_of_agi.debug.events import DebugEvent, FrameTurnCompleted, ModelCallCompleted

_MODEL_TIMING_ROLES = (
    "agent",
    "change",
    "memory",
    "world",
    "goal",
    "reward_judge",
)


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
        self._min_turn_seconds: float | None = None
        self._max_turn_seconds: float | None = None
        self._model_call_totals = {role: 0.0 for role in _MODEL_TIMING_ROLES}
        self._model_call_counts = {role: 0 for role in _MODEL_TIMING_ROLES}
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
            duration = max(0.0, float(event.turn_duration_seconds))
            if self._started_at is None:
                self._started_at = now - duration

            self._turn_count += 1
            if event.controllable:
                self._controllable_turn_count += 1
            self._total_turn_seconds += duration
            self._min_turn_seconds = (
                duration
                if self._min_turn_seconds is None
                else min(self._min_turn_seconds, duration)
            )
            self._max_turn_seconds = (
                duration
                if self._max_turn_seconds is None
                else max(self._max_turn_seconds, duration)
            )
            self._completed_levels_by_game[
                (event.game_index, event.game_id)
            ] = event.completed_levels

            if self._turn_count % self.selected_game_count != 0:
                return

            self.output.write(self._summary_line(now) + "\n")
            self.output.flush()

    def _record_model_call(self, event: ModelCallCompleted) -> None:
        role = event.role
        if role not in self._model_call_totals:
            return
        with self._lock:
            self._model_call_totals[role] += max(
                0.0,
                float(event.duration_seconds),
            )
            self._model_call_counts[role] += 1

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
        total_completed_levels = sum(self._completed_levels_by_game.values())
        avg_controllable_turns_per_game = (
            self._controllable_turn_count / self.selected_game_count
        )
        line = (
            "throughput:"
            f" turns={self._turn_count}"
            f" games={self.selected_game_count}"
            f" avg_controllable_turns_per_game={avg_controllable_turns_per_game:.2f}"
            f" avg_turn_sec={avg_turn_seconds:.3f}"
            f" min_turn_sec={(self._min_turn_seconds or 0.0):.3f}"
            f" max_turn_sec={(self._max_turn_seconds or 0.0):.3f}"
            f" turns_per_min={turns_per_minute:.2f}"
            f" total_completed_levels={total_completed_levels}"
        )
        for role in _MODEL_TIMING_ROLES:
            count = self._model_call_counts[role]
            average = (
                self._model_call_totals[role] / count
                if count
                else 0.0
            )
            line += f" avg_model_sec_{role}={average:.3f}"
        return line
