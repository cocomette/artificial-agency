"""Tests for the shared model-call scheduler."""

from __future__ import annotations

import threading
import time

import pytest

from face_of_agi.models.providers.scheduler import (
    ModelCallRuntimeContext,
    ModelCallScheduler,
    ModelSchedulerTimeoutError,
)


def _context(game_id: str, role: str = "agent") -> ModelCallRuntimeContext:
    return ModelCallRuntimeContext(
        run_id="run-1",
        game_id=game_id,
        turn_id=1,
        role=role,
    )


def test_scheduler_skips_ineligible_same_game_head_of_line_call() -> None:
    scheduler = ModelCallScheduler(
        max_concurrent_calls=2,
        max_concurrent_calls_per_game=1,
    )
    release_first = threading.Event()
    first_started = threading.Event()
    second_same_game_started = threading.Event()
    other_game_started = threading.Event()

    def run_first() -> None:
        scheduler.run(
            lambda: (first_started.set(), release_first.wait(timeout=2)),
            context=_context("game-a"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=1,
            request_timeout_seconds=1,
        )

    def run_second_same_game() -> None:
        scheduler.run(
            lambda: second_same_game_started.set(),
            context=_context("game-a"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=2,
            request_timeout_seconds=2,
        )

    def run_other_game() -> None:
        scheduler.run(
            lambda: other_game_started.set(),
            context=_context("game-b"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=1,
            request_timeout_seconds=1,
        )

    first = threading.Thread(target=run_first)
    same_game = threading.Thread(target=run_second_same_game)
    other_game = threading.Thread(target=run_other_game)
    first.start()
    assert first_started.wait(timeout=1)
    same_game.start()
    time.sleep(0.05)
    other_game.start()

    assert other_game_started.wait(timeout=1)
    assert not second_same_game_started.is_set()

    release_first.set()
    first.join(timeout=2)
    same_game.join(timeout=2)
    other_game.join(timeout=2)
    assert second_same_game_started.is_set()


def test_scheduler_global_capacity_blocks_until_release() -> None:
    scheduler = ModelCallScheduler(
        max_concurrent_calls=1,
        max_concurrent_calls_per_game=1,
    )
    release_first = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()

    first = threading.Thread(
        target=lambda: scheduler.run(
            lambda: (first_started.set(), release_first.wait(timeout=2)),
            context=_context("game-a"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=1,
            request_timeout_seconds=1,
        )
    )
    second = threading.Thread(
        target=lambda: scheduler.run(
            lambda: second_started.set(),
            context=_context("game-b"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=2,
            request_timeout_seconds=2,
        )
    )

    first.start()
    assert first_started.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert second_started.is_set()


def test_scheduler_queue_timeout_raises_model_scheduler_timeout() -> None:
    scheduler = ModelCallScheduler(
        max_concurrent_calls=1,
        max_concurrent_calls_per_game=1,
    )
    release_first = threading.Event()
    first_started = threading.Event()
    first = threading.Thread(
        target=lambda: scheduler.run(
            lambda: (first_started.set(), release_first.wait(timeout=2)),
            context=_context("game-a"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=1,
            request_timeout_seconds=1,
        )
    )
    first.start()
    assert first_started.wait(timeout=1)

    with pytest.raises(ModelSchedulerTimeoutError):
        scheduler.run(
            lambda: None,
            context=_context("game-b"),
            provider="vllm",
            model="model",
            queue_timeout_seconds=0.01,
            request_timeout_seconds=1,
        )

    release_first.set()
    first.join(timeout=2)
