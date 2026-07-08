"""Tests for parallel multi-game runtime support."""

from __future__ import annotations

from io import StringIO
import json
import threading
import time

import pytest

from face_of_agi.contracts import (
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    GameRunResult,
    Observation,
    ObservationRef,
    RuntimeConfig,
)
from face_of_agi.environment.config import (
    EnvironmentConfig,
    load_environment_config,
)
from face_of_agi.debug.events import FrameTurnCompleted, ModelCallCompleted
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.runtime.parallel import ParallelGameRunSpec, ParallelRuntimeLoop
from face_of_agi.runtime import shell


def _minimal_config_text(*, selection: str) -> str:
    return "\n".join(
        [
            selection,
            "max_actions_per_level: 1",
            "models:",
            "  shared_vlm:",
            "    backend: vllm",
            "    model: qwen",
            "  agent:",
            "    backend: vllm",
            "  change:",
            "    backend: vllm",
            "  memory:",
            "    backend: vllm",
            "  world:",
            "    backend: vllm",
            "  goal:",
            "    backend: vllm",
            "  interest:",
            "    backend: vllm",
            "  reward_judge:",
            "    backend: vllm",
        ]
    )


def _spec(
    *,
    game_index: int,
    game_id: str,
    run_id: str,
    database_path,
) -> ParallelGameRunSpec:
    return ParallelGameRunSpec(
        game_index=game_index,
        game_id=game_id,
        run_id=run_id,
        database_path=database_path,
        environment_config=EnvironmentConfig(
            game_index=game_index,
            game_id=game_id,
            max_actions_per_level=1,
        ),
    )


def _result(spec: ParallelGameRunSpec) -> GameRunResult:
    return GameRunResult(
        run_id=spec.run_id,
        game_id=spec.game_id,
        stop_reason="action_limit_reached",
        step_count=1,
    )


def test_environment_config_loads_parallel_game_indices(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_indices: [1, 2, 3]")
        + "\nmax_parallel_games: 2\n",
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.game_index is None
    assert config.game_indices == (1, 2, 3)
    assert config.game_ids == ()
    assert config.game_selection is None
    assert config.max_parallel_games == 2
    assert config.max_game_retries == 0


def test_environment_config_loads_max_game_retries(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_indices: [1, 2]")
        + "\nmax_game_retries: 1\n",
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.max_game_retries == 1


def test_environment_config_loads_level_cap(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_indices: [1, 2]")
        + "\nmax_levels_per_game: 1\n",
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.max_levels_per_game == 1


def test_environment_config_rejects_removed_game_action_cap(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_indices: [1, 2]")
        + "\nmax_actions_per_game: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_actions_per_game has been removed"):
        load_environment_config(config_path)


def test_environment_config_loads_explicit_game_ids(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_ids: [ls20-abc, vc33-def]")
        + "\nmax_parallel_games: 2\n",
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.game_index is None
    assert config.game_indices == ()
    assert config.game_ids == ("ls20-abc", "vc33-def")
    assert config.game_selection is None


def test_environment_config_loads_all_available_selection(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_selection: all_available"),
        encoding="utf-8",
    )

    config = load_environment_config(config_path)

    assert config.game_index is None
    assert config.game_indices == ()
    assert config.game_ids == ()
    assert config.game_selection == "all_available"


def test_parallel_runtime_resolves_all_available_from_local_catalog(tmp_path) -> None:
    catalog_path = tmp_path / "local_games.json"
    catalog_path.write_text(
        json.dumps({"10": "game-10", "2": "game-2"}) + "\n",
        encoding="utf-8",
    )
    config = EnvironmentConfig(
        max_actions_per_level=1,
        game_selection="all_available",
        game_catalog_path=str(catalog_path),
    )

    assert shell._resolve_selected_game_ids(config) == (
        (2, "game-2"),
        (10, "game-10"),
    )


def test_environment_config_rejects_multiple_game_selectors(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        "\n".join(
            [
                "game_index: 0",
                "game_indices: [1, 2]",
                "max_actions_per_level: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_environment_config(config_path)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("game_indices: []", "game_indices must not be empty"),
        ("game_indices: [1, 1]", "game_indices must not contain duplicates"),
        ("game_indices: [-1]", "game_indices must be non-negative"),
        ("game_ids: []", "game_ids must not be empty"),
        ("game_ids: [ls20, ls20]", "game_ids must not contain duplicates"),
        ("game_selection: public", "game_selection must be all_available"),
        ("game_indices: [1]\nmax_parallel_games: 0", "max_parallel_games"),
        ("game_indices: [1]\nmax_game_retries: -1", "max_game_retries"),
        ("game_indices: [1]\ngame_id: game-1", "game_id cannot be set"),
        ("game_ids: [ls20]\ngame_id: game-1", "game_id cannot be set"),
        (
            "game_selection: all_available\ngame_id: game-1",
            "game_id cannot be set",
        ),
    ],
)
def test_environment_config_rejects_invalid_parallel_selection(
    tmp_path,
    selection: str,
    message: str,
) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection=selection),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_environment_config(config_path)


def test_environment_config_requires_one_game_selection(tmp_path) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        "\n".join(["max_actions_per_level: 1"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="game_index, game_indices, game_ids"):
        load_environment_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_levels_per_game", 0),
        ("max_levels_per_game", -1),
    ],
)
def test_environment_config_rejects_invalid_level_caps(
    tmp_path,
    field: str,
    value: int,
) -> None:
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        _minimal_config_text(selection="game_indices: [1]")
        + f"\n{field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_environment_config(config_path)


def test_parallel_runtime_runs_games_concurrently_with_distinct_databases(
    tmp_path,
) -> None:
    barrier = threading.Barrier(2)

    def run_game(spec: ParallelGameRunSpec, trace_output) -> GameRunResult:
        del trace_output
        StateMemory(SQLiteDatabase(spec.database_path))
        barrier.wait(timeout=2)
        return _result(spec)

    specs = (
        _spec(
            game_index=1,
            game_id="game-1",
            run_id="run-1",
            database_path=tmp_path / "one.sqlite",
        ),
        _spec(
            game_index=2,
            game_id="game-2",
            run_id="run-2",
            database_path=tmp_path / "two.sqlite",
        ),
    )

    result = ParallelRuntimeLoop(run_game, trace_output=StringIO()).run(
        batch_run_id="batch-1",
        specs=specs,
        max_parallel_games=2,
    )

    assert [success.game_index for success in result.successes] == [1, 2]
    assert result.failures == ()
    assert (tmp_path / "one.sqlite").exists()
    assert (tmp_path / "two.sqlite").exists()


def test_parallel_runtime_continues_after_game_failure(tmp_path) -> None:
    def run_game(spec: ParallelGameRunSpec, trace_output) -> GameRunResult:
        del trace_output
        if spec.game_index == 2:
            raise RuntimeError("boom")
        return _result(spec)

    specs = (
        _spec(
            game_index=1,
            game_id="game-1",
            run_id="run-1",
            database_path=tmp_path / "one.sqlite",
        ),
        _spec(
            game_index=2,
            game_id="game-2",
            run_id="run-2",
            database_path=tmp_path / "two.sqlite",
        ),
    )

    result = ParallelRuntimeLoop(run_game, trace_output=StringIO()).run(
        batch_run_id="batch-1",
        specs=specs,
        max_parallel_games=2,
    )

    assert [success.game_index for success in result.successes] == [1]
    assert [failure.game_index for failure in result.failures] == [2]
    assert result.failures[0].exception_type == "RuntimeError"
    assert result.failures[0].message == "boom"
    assert result.failures[0].attempt_count == 1


def test_parallel_runtime_retries_game_and_reports_success_attempt(
    tmp_path,
) -> None:
    attempts: dict[str, int] = {}
    seen_specs: list[ParallelGameRunSpec] = []

    def run_game(spec: ParallelGameRunSpec, trace_output) -> GameRunResult:
        del trace_output
        seen_specs.append(spec)
        attempts[spec.game_id] = attempts.get(spec.game_id, 0) + 1
        if spec.game_id == "game-2" and attempts[spec.game_id] == 1:
            raise RuntimeError("bad json")
        return _result(spec)

    specs = (
        _spec(
            game_index=1,
            game_id="game-1",
            run_id="run-1",
            database_path=tmp_path / "one.sqlite",
        ),
        _spec(
            game_index=2,
            game_id="game-2",
            run_id="run-2",
            database_path=tmp_path / "two.sqlite",
        ),
    )

    result = ParallelRuntimeLoop(run_game, trace_output=StringIO()).run(
        batch_run_id="batch-1",
        specs=specs,
        max_parallel_games=2,
        max_game_retries=1,
    )

    assert result.failures == ()
    assert [(success.game_id, success.attempt_count) for success in result.successes] == [
        ("game-1", 1),
        ("game-2", 2),
    ]
    retry_success = result.successes[1]
    assert retry_success.run_id == "run-2-retry-1"
    assert retry_success.database_path.endswith("two-retry-1.sqlite")
    assert retry_success.result.run_id == "run-2-retry-1"
    assert [spec.attempt_index for spec in seen_specs if spec.game_id == "game-2"] == [
        0,
        1,
    ]


def test_parallel_runtime_records_final_failure_after_retry_exhaustion(
    tmp_path,
) -> None:
    def run_game(spec: ParallelGameRunSpec, trace_output) -> GameRunResult:
        del spec, trace_output
        raise RuntimeError("context length exceeded")

    specs = (
        _spec(
            game_index=1,
            game_id="game-1",
            run_id="run-1",
            database_path=tmp_path / "one.sqlite",
        ),
    )

    result = ParallelRuntimeLoop(run_game, trace_output=StringIO()).run(
        batch_run_id="batch-1",
        specs=specs,
        max_parallel_games=1,
        max_game_retries=1,
    )

    assert result.successes == ()
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.attempt_count == 2
    assert failure.run_id == "run-1-retry-1"
    assert failure.database_path.endswith("one-retry-1.sqlite")
    assert failure.exception_type == "RuntimeError"
    assert failure.message == "context length exceeded"


def test_parallel_runtime_does_not_retry_after_deadline(
    tmp_path,
) -> None:
    attempts = 0

    def run_game(spec: ParallelGameRunSpec, trace_output) -> GameRunResult:
        nonlocal attempts
        del spec, trace_output
        attempts += 1
        raise RuntimeError("boom")

    specs = (
        _spec(
            game_index=1,
            game_id="game-1",
            run_id="run-1",
            database_path=tmp_path / "one.sqlite",
        ),
    )

    result = ParallelRuntimeLoop(run_game, trace_output=StringIO()).run(
        batch_run_id="batch-1",
        specs=specs,
        max_parallel_games=1,
        max_game_retries=1,
        deadline_monotonic=time.monotonic() - 1,
    )

    assert attempts == 1
    assert result.successes == ()
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.attempt_count == 1
    assert failure.run_id == "run-1"
    assert failure.database_path.endswith("one.sqlite")


def test_shell_builds_isolated_parallel_specs(monkeypatch, tmp_path) -> None:
    catalog_path = tmp_path / "games.json"
    catalog_path.write_text(json.dumps({"1": "game-a", "2": "game-b"}))
    captured_specs: list[ParallelGameRunSpec] = []
    lock = threading.Lock()

    def fake_run_game(
        spec: ParallelGameRunSpec,
        trace_output,
    ) -> GameRunResult:
        del trace_output
        with lock:
            captured_specs.append(spec)
        return _result(spec)

    monkeypatch.setattr(shell, "_build_parallel_batch_run_id", lambda: "parallel-test")
    monkeypatch.setattr(shell, "_run_parallel_game", fake_run_game)

    config = EnvironmentConfig(
        game_indices=(1, 2),
        max_parallel_games=1,
        max_actions_per_level=1,
        game_catalog_path=str(catalog_path),
    )

    result = shell._run_parallel_config(
        base_environment_config=config,
        base_database_path=tmp_path / "memory.sqlite",
        trace_output=StringIO(),
    )

    assert [success.run_id for success in result.successes] == [
        "parallel-test-game-index-1",
        "parallel-test-game-index-2",
    ]
    assert [spec.game_id for spec in captured_specs] == ["game-a", "game-b"]
    assert [spec.environment_config.game_index for spec in captured_specs] == [1, 2]
    assert [spec.environment_config.game_indices for spec in captured_specs] == [
        (),
        (),
    ]
    assert [spec.database_path.name for spec in captured_specs] == [
        "memory-game-index-1.sqlite",
        "memory-game-index-2.sqlite",
    ]


def test_shell_parallel_config_shares_live_turn_monitor(monkeypatch, tmp_path) -> None:
    catalog_path = tmp_path / "games.json"
    catalog_path.write_text(json.dumps({"1": "game-a", "2": "game-b"}))
    captured_specs: list[ParallelGameRunSpec] = []
    lock = threading.Lock()

    def fake_run_game(
        spec: ParallelGameRunSpec,
        trace_output,
    ) -> GameRunResult:
        del trace_output
        with lock:
            captured_specs.append(spec)
        assert spec.live_turn_monitor is not None
        spec.live_turn_monitor.emit(
            ModelCallCompleted(role="agent", duration_seconds=1.0)
        )
        spec.live_turn_monitor.emit(
            ModelCallCompleted(role="change", duration_seconds=2.0)
        )
        spec.live_turn_monitor.emit(
            FrameTurnCompleted(
                run_id=spec.run_id,
                game_id=spec.game_id,
                game_index=spec.game_index,
                turn_id=1,
                env_step=0,
                frame_index=0,
                frame_count=1,
                controllable=True,
                action=ActionSpec(action_id="ACTION1"),
                turn_duration_seconds=float(spec.game_index),
                completed_levels=spec.game_index,
                remaining_actions=0,
            )
        )
        return _result(spec)

    monkeypatch.setattr(shell, "_build_parallel_batch_run_id", lambda: "parallel-test")
    monkeypatch.setattr(shell, "_run_parallel_game", fake_run_game)

    config = EnvironmentConfig(
        game_indices=(1, 2),
        max_parallel_games=1,
        max_actions_per_level=1,
        game_catalog_path=str(catalog_path),
        live_turn_monitor=True,
    )
    output = StringIO()

    shell._run_parallel_config(
        base_environment_config=config,
        base_database_path=tmp_path / "memory.sqlite",
        trace_output=output,
    )

    monitors = {id(spec.live_turn_monitor) for spec in captured_specs}
    assert len(monitors) == 1
    rendered = output.getvalue()
    assert rendered.count("throughput:") == 1
    assert "turns=2" in rendered
    assert "games=2" in rendered
    assert "avg_turn_sec=1.500" in rendered
    assert "avg_controllable_turns_per_game=1.00" in rendered
    assert "avg_model_sec_agent=1.000" in rendered
    assert "avg_model_sec_change=2.000" in rendered
    assert "avg_model_sec_memory=0.000" in rendered
    assert "avg_model_sec_world=0.000" in rendered
    assert "avg_model_sec_goal=0.000" in rendered
    assert "avg_model_sec_reward_judge=0.000" in rendered
    assert "total_completed_levels=3" in rendered


def test_shell_database_path_for_game_uses_custom_suffix(tmp_path) -> None:
    assert shell._database_path_for_game(
        tmp_path / "custom.db",
        3,
    ) == tmp_path / "custom-game-index-3.db"
