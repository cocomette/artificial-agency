"""Parallel runtime helper for isolated multi-game shell runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import threading
import time
from typing import Any, TextIO

from face_of_agi.contracts import (
    GameRunResult,
    ParallelGameRunFailure,
    ParallelGameRunResult,
    ParallelGameRunSuccess,
)
from face_of_agi.environment.config import EnvironmentConfig


@dataclass(frozen=True, slots=True)
class ParallelGameRunSpec:
    """Resolved single-game run input inside a parallel runtime batch."""

    game_index: int
    game_id: str
    run_id: str
    database_path: Path
    environment_config: EnvironmentConfig
    arc_environment: Any | None = None
    live_turn_monitor: Any | None = None
    online_lora_manager: Any | None = None
    attempt_index: int = 0
    deadline_monotonic: float | None = None


ParallelGameRunner = Callable[[ParallelGameRunSpec, TextIO], GameRunResult]
ParallelRetrySpecFactory = Callable[[ParallelGameRunSpec, int], ParallelGameRunSpec]


@dataclass(frozen=True, slots=True)
class _ParallelGameOutcome:
    spec: ParallelGameRunSpec
    attempt_count: int
    result: GameRunResult | None = None
    exception_type: str | None = None
    message: str | None = None


class LockedTextIO:
    """TextIO proxy that serializes writes from parallel game workers."""

    def __init__(self, target: TextIO, lock: threading.Lock) -> None:
        self._target = target
        self._lock = lock

    @property
    def encoding(self) -> str | None:
        return getattr(self._target, "encoding", None)

    def write(self, text: str) -> int:
        with self._lock:
            return self._target.write(text)

    def flush(self) -> None:
        with self._lock:
            self._target.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._target, "isatty", lambda: False)())

    def writable(self) -> bool:
        return bool(getattr(self._target, "writable", lambda: True)())

    def fileno(self) -> int:
        return int(getattr(self._target, "fileno")())


class ParallelRuntimeLoop:
    """Run multiple isolated game loops concurrently and aggregate outcomes."""

    def __init__(
        self,
        run_game: ParallelGameRunner,
        *,
        trace_output: TextIO | None = None,
    ) -> None:
        self.run_game = run_game
        self.trace_output = trace_output or sys.stdout

    def run(
        self,
        *,
        batch_run_id: str,
        specs: Sequence[ParallelGameRunSpec],
        max_parallel_games: int | None = None,
        max_game_retries: int = 0,
        retry_spec_factory: ParallelRetrySpecFactory | None = None,
        deadline_monotonic: float | None = None,
    ) -> ParallelGameRunResult:
        """Run the selected games concurrently, continuing after failures."""

        if not specs:
            raise ValueError("parallel runtime requires at least one game")
        if max_parallel_games is not None and max_parallel_games < 1:
            raise ValueError("max_parallel_games must be at least 1")
        if max_game_retries < 0:
            raise ValueError("max_game_retries must be non-negative")

        max_workers = max_parallel_games or len(specs)
        output = LockedTextIO(self.trace_output, threading.Lock())
        successes: dict[int, ParallelGameRunSuccess] = {}
        failures: dict[int, ParallelGameRunFailure] = {}
        build_retry_spec = retry_spec_factory or retry_parallel_game_spec

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_specs = {
                executor.submit(
                    self._run_game_with_retries,
                    spec,
                    output,
                    max_game_retries,
                    build_retry_spec,
                    deadline_monotonic,
                ): (index, spec)
                for index, spec in enumerate(specs)
            }
            for future in as_completed(future_specs):
                index, spec = future_specs[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    failures[index] = ParallelGameRunFailure(
                        game_index=spec.game_index,
                        game_id=spec.game_id,
                        run_id=spec.run_id,
                        database_path=str(spec.database_path),
                        exception_type=type(exc).__name__,
                        message=str(exc),
                        attempt_count=1,
                    )
                    continue

                if outcome.result is None:
                    failures[index] = ParallelGameRunFailure(
                        game_index=outcome.spec.game_index,
                        game_id=outcome.spec.game_id,
                        run_id=outcome.spec.run_id,
                        database_path=str(outcome.spec.database_path),
                        exception_type=outcome.exception_type or "Exception",
                        message=outcome.message or "",
                        attempt_count=outcome.attempt_count,
                    )
                    continue

                successes[index] = ParallelGameRunSuccess(
                    game_index=outcome.spec.game_index,
                    game_id=outcome.spec.game_id,
                    run_id=outcome.spec.run_id,
                    database_path=str(outcome.spec.database_path),
                    result=outcome.result,
                    attempt_count=outcome.attempt_count,
                )

        ordered_indices = range(len(specs))
        return ParallelGameRunResult(
            batch_run_id=batch_run_id,
            successes=tuple(
                successes[index]
                for index in ordered_indices
                if index in successes
            ),
            failures=tuple(
                failures[index]
                for index in ordered_indices
                if index in failures
            ),
        )

    def _run_game_with_retries(
        self,
        spec: ParallelGameRunSpec,
        output: TextIO,
        max_game_retries: int,
        retry_spec_factory: ParallelRetrySpecFactory,
        deadline_monotonic: float | None,
    ) -> _ParallelGameOutcome:
        """Run one game, retrying with isolated run/db ids after failures."""

        active_spec = spec
        for attempt_index in range(max_game_retries + 1):
            try:
                active_spec = (
                    spec
                    if attempt_index == 0
                    else retry_spec_factory(spec, attempt_index)
                )
                return _ParallelGameOutcome(
                    spec=active_spec,
                    attempt_count=attempt_index + 1,
                    result=self.run_game(active_spec, output),
                )
            except Exception as exc:
                if attempt_index >= max_game_retries or _deadline_expired(
                    deadline_monotonic
                ):
                    return _ParallelGameOutcome(
                        spec=active_spec,
                        attempt_count=attempt_index + 1,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                output.write(
                    "retrying:"
                    f" game_index={spec.game_index}"
                    f" game_id={spec.game_id}"
                    f" attempt={attempt_index + 1}"
                    f" next_attempt={attempt_index + 2}"
                    f" error={type(exc).__name__}: {exc}\n"
                )
                output.flush()


def retry_parallel_game_spec(
    spec: ParallelGameRunSpec,
    attempt_index: int,
) -> ParallelGameRunSpec:
    """Return an isolated retry spec for a non-initial attempt."""

    if attempt_index < 1:
        raise ValueError("retry attempt_index must be at least 1")
    return replace(
        spec,
        run_id=f"{spec.run_id}-retry-{attempt_index}",
        database_path=_retry_database_path(spec.database_path, attempt_index),
        arc_environment=None,
        attempt_index=attempt_index,
    )


def _retry_database_path(path: Path, attempt_index: int) -> Path:
    return path.with_name(f"{path.stem}-retry-{attempt_index}{path.suffix}")


def _deadline_expired(deadline_monotonic: float | None) -> bool:
    return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic
