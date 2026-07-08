"""Shared model-call scheduler and runtime context."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic, perf_counter
import threading
from typing import Any, TypeVar

T = TypeVar("T")
ModelCallEventSink = Callable[..., None]


@dataclass(frozen=True, slots=True)
class ModelCallRuntimeContext:
    """Run/game/turn metadata for one model-role call."""

    run_id: str
    game_id: str
    turn_id: int | None
    role: str
    emit_event: ModelCallEventSink | None = None


class ModelSchedulerTimeoutError(TimeoutError):
    """Raised when a queued model call cannot start in time."""


@dataclass(slots=True)
class _QueueEntry:
    game_id: str
    enqueued_at: float
    admitted: bool = False


class ModelCallScheduler:
    """Strict-FIFO shared scheduler with a per-game in-flight cap."""

    def __init__(
        self,
        *,
        max_concurrent_calls: int,
        max_concurrent_calls_per_game: int,
    ) -> None:
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1")
        if max_concurrent_calls_per_game < 1:
            raise ValueError("max_concurrent_calls_per_game must be at least 1")
        self.max_concurrent_calls = max_concurrent_calls
        self.max_concurrent_calls_per_game = max_concurrent_calls_per_game
        self._condition = threading.Condition()
        self._queue: deque[_QueueEntry] = deque()
        self._active_total = 0
        self._active_by_game: dict[str, int] = {}

    def run(
        self,
        fn: Callable[[], T],
        *,
        context: ModelCallRuntimeContext,
        provider: str,
        model: str | None,
        queue_timeout_seconds: float | None,
        request_timeout_seconds: float | None,
    ) -> T:
        """Run `fn` after this call is admitted by the scheduler."""

        queue_wait_seconds = self._acquire(
            game_id=context.game_id,
            timeout_seconds=queue_timeout_seconds,
            context=context,
            provider=provider,
            model=model,
            request_timeout_seconds=request_timeout_seconds,
        )
        try:
            return _run_provider_call(
                fn,
                context=context,
                provider=provider,
                model=model,
                queue_wait_seconds=queue_wait_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
        finally:
            self._release(context.game_id)

    def _acquire(
        self,
        *,
        game_id: str,
        timeout_seconds: float | None,
        context: ModelCallRuntimeContext,
        provider: str,
        model: str | None,
        request_timeout_seconds: float | None,
    ) -> float:
        entry = _QueueEntry(game_id=game_id, enqueued_at=perf_counter())
        deadline = None if timeout_seconds is None else monotonic() + timeout_seconds
        _emit_model_call_event(
            context,
            provider=provider,
            model=model,
            event="queue_enter",
            status="queued",
            timeout_seconds=request_timeout_seconds,
            metadata={"queue_timeout_seconds": timeout_seconds},
        )
        with self._condition:
            self._queue.append(entry)
            while True:
                self._admit_eligible_locked()
                if entry.admitted:
                    wait_seconds = perf_counter() - entry.enqueued_at
                    _emit_model_call_event(
                        context,
                        provider=provider,
                        model=model,
                        event="queue_start",
                        status="started",
                        queue_wait_seconds=wait_seconds,
                        timeout_seconds=request_timeout_seconds,
                    )
                    return wait_seconds
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._remove_entry_locked(entry)
                    wait_seconds = perf_counter() - entry.enqueued_at
                    _emit_model_call_event(
                        context,
                        provider=provider,
                        model=model,
                        event="queue_timeout",
                        status="timeout",
                        queue_wait_seconds=wait_seconds,
                        timeout_seconds=request_timeout_seconds,
                        metadata={"queue_timeout_seconds": timeout_seconds},
                    )
                    raise ModelSchedulerTimeoutError(
                        f"model call for game {game_id!r} role {context.role!r} "
                        f"waited {wait_seconds:.3f}s for scheduler capacity"
                    )
                self._condition.wait(timeout=remaining)

    def _release(self, game_id: str) -> None:
        with self._condition:
            self._active_total -= 1
            active_for_game = self._active_by_game.get(game_id, 0) - 1
            if active_for_game > 0:
                self._active_by_game[game_id] = active_for_game
            else:
                self._active_by_game.pop(game_id, None)
            self._admit_eligible_locked()
            self._condition.notify_all()

    def _admit_eligible_locked(self) -> None:
        while self._active_total < self.max_concurrent_calls:
            index = self._next_eligible_index_locked()
            if index is None:
                return
            entry = self._queue[index]
            del self._queue[index]
            entry.admitted = True
            self._active_total += 1
            self._active_by_game[entry.game_id] = (
                self._active_by_game.get(entry.game_id, 0) + 1
            )
            self._condition.notify_all()

    def _next_eligible_index_locked(self) -> int | None:
        for index, entry in enumerate(self._queue):
            active_for_game = self._active_by_game.get(entry.game_id, 0)
            if active_for_game < self.max_concurrent_calls_per_game:
                return index
        return None

    def _remove_entry_locked(self, entry: _QueueEntry) -> None:
        try:
            self._queue.remove(entry)
        except ValueError:
            return
        self._condition.notify_all()


_MODEL_CALL_CONTEXT: ContextVar[ModelCallRuntimeContext | None] = ContextVar(
    "face_of_agi_model_call_context",
    default=None,
)


@contextmanager
def model_call_context(
    *,
    run_id: str,
    game_id: str,
    turn_id: int | None,
    role: str,
    emit_event: ModelCallEventSink | None = None,
) -> Iterator[None]:
    """Attach run/game/turn metadata to model provider calls in this context."""

    token = _MODEL_CALL_CONTEXT.set(
        ModelCallRuntimeContext(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            role=role,
            emit_event=emit_event,
        )
    )
    try:
        yield
    finally:
        _MODEL_CALL_CONTEXT.reset(token)


def current_model_call_context() -> ModelCallRuntimeContext | None:
    """Return the active model-call context, if orchestration set one."""

    return _MODEL_CALL_CONTEXT.get()


def run_unscheduled_provider_call(
    fn: Callable[[], T],
    *,
    context: ModelCallRuntimeContext | None,
    provider: str,
    model: str | None,
    request_timeout_seconds: float | None,
) -> T:
    """Run a provider call with lifecycle telemetry but no scheduler."""

    if context is None:
        return fn()
    return _run_provider_call(
        fn,
        context=context,
        provider=provider,
        model=model,
        queue_wait_seconds=None,
        request_timeout_seconds=request_timeout_seconds,
    )


def _run_provider_call(
    fn: Callable[[], T],
    *,
    context: ModelCallRuntimeContext,
    provider: str,
    model: str | None,
    queue_wait_seconds: float | None,
    request_timeout_seconds: float | None,
) -> T:
    started_at = perf_counter()
    _emit_model_call_event(
        context,
        provider=provider,
        model=model,
        event="provider_start",
        status="started",
        queue_wait_seconds=queue_wait_seconds,
        timeout_seconds=request_timeout_seconds,
    )
    try:
        result = fn()
    except Exception as exc:
        duration_seconds = perf_counter() - started_at
        _emit_model_call_event(
            context,
            provider=provider,
            model=model,
            event="provider_error",
            status=_status_for_exception(exc),
            queue_wait_seconds=queue_wait_seconds,
            duration_seconds=duration_seconds,
            timeout_seconds=request_timeout_seconds,
            metadata={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    duration_seconds = perf_counter() - started_at
    _emit_model_call_event(
        context,
        provider=provider,
        model=model,
        event="provider_end",
        status="success",
        queue_wait_seconds=queue_wait_seconds,
        duration_seconds=duration_seconds,
        timeout_seconds=request_timeout_seconds,
    )
    return result


def _emit_model_call_event(
    context: ModelCallRuntimeContext,
    *,
    provider: str,
    model: str | None,
    event: str,
    status: str,
    queue_wait_seconds: float | None = None,
    duration_seconds: float | None = None,
    timeout_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if context.emit_event is None:
        return
    context.emit_event(
        run_id=context.run_id,
        game_id=context.game_id,
        turn_id=context.turn_id,
        role=context.role,
        provider=provider,
        model=model,
        event=event,
        status=status,
        queue_wait_seconds=queue_wait_seconds,
        duration_seconds=duration_seconds,
        timeout_seconds=timeout_seconds,
        metadata=metadata,
    )


def _status_for_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    return "error"
