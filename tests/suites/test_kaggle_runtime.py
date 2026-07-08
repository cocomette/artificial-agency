"""Tests for the Kaggle runtime entrypoint without calling Kaggle."""

from __future__ import annotations

from io import StringIO

from arc_agi import OperationMode

from face_of_agi.contracts import (
    GameRunResult,
    ParallelGameRunFailure,
    ParallelGameRunResult,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.runtime import kaggle
from face_of_agi.runtime.parallel import ParallelGameRunSpec


def _minimal_config_text(selection: str) -> str:
    return "\n".join(
        [
            selection,
            "max_actions_per_level: 1",
            "models:",
            "  change:",
            "    backend: openai",
            "    model: gpt-5-nano",
            "  compacter:",
            "    backend: openai",
            "    model: gpt-5-nano",
            "  updater:",
            "    agent:",
            "      backend: openai",
            "      model: gpt-5-nano",
        ]
    )


class _FakeGame:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id


class _FakeArcade:
    def __init__(self) -> None:
        self.closed: list[str] = []
        self.made: list[tuple[str, str]] = []
        self.games = (_FakeGame("ls20-aaa"), _FakeGame("vc33-bbb"))

    def list_available_game_ids(self):
        return tuple(game.game_id for game in self.games)

    def open_scorecard(self, *, tags):
        self.tags = tags
        return "scorecard-1"

    def close_scorecard(self, card_id):
        self.closed.append(card_id)
        return {"card_id": card_id}

    def make_scorecard_environment(self, game_id, *, scorecard_id):
        self.made.append((game_id, scorecard_id))
        return object()


def _result(spec: ParallelGameRunSpec) -> GameRunResult:
    return GameRunResult(
        run_id=spec.run_id,
        game_id=spec.game_id,
        stop_reason="action_limit_reached",
        step_count=1,
    )


def test_kaggle_runner_discovers_all_games_and_closes_scorecard(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "kaggle.yaml"
    config_path.write_text(
        _minimal_config_text("game_selection: all_available"),
        encoding="utf-8",
    )
    fake_arcade = _FakeArcade()
    captured_specs: list[ParallelGameRunSpec] = []

    def fake_run_game(spec: ParallelGameRunSpec, trace_output):
        del trace_output
        captured_specs.append(spec)
        return _result(spec)

    monkeypatch.setattr(kaggle, "_run_kaggle_game", fake_run_game)
    monkeypatch.setattr(kaggle, "_build_kaggle_batch_run_id", lambda: "kaggle-test")

    result = kaggle.run_config(
        config_path=config_path,
        database_dir=tmp_path / "runs",
        tags=("unit",),
        arcade=fake_arcade,
        trace_output=StringIO(),
    )

    assert [success.game_id for success in result.successes] == [
        "ls20-aaa",
        "vc33-bbb",
    ]
    assert result.failures == ()
    assert fake_arcade.tags == ("unit",)
    assert fake_arcade.closed == ["scorecard-1"]
    assert fake_arcade.made == [
        ("ls20-aaa", "scorecard-1"),
        ("vc33-bbb", "scorecard-1"),
    ]
    assert [spec.database_path.name for spec in captured_specs] == [
        "memory-ls20-aaa.sqlite",
        "memory-vc33-bbb.sqlite",
    ]
    assert [spec.environment_config.game_id for spec in captured_specs] == [
        "ls20-aaa",
        "vc33-bbb",
    ]
    assert [spec.environment_config.operation_mode for spec in captured_specs] == [
        OperationMode.COMPETITION,
        OperationMode.COMPETITION,
    ]
    assert [spec.environment_config.game_selection for spec in captured_specs] == [
        None,
        None,
    ]


def test_kaggle_runner_closes_scorecard_after_worker_failure(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "kaggle.yaml"
    config_path.write_text(
        _minimal_config_text("game_selection: all_available"),
        encoding="utf-8",
    )
    fake_arcade = _FakeArcade()

    def fake_run_game(spec: ParallelGameRunSpec, trace_output):
        del trace_output
        if spec.game_id == "vc33-bbb":
            raise RuntimeError("boom")
        return _result(spec)

    monkeypatch.setattr(kaggle, "_run_kaggle_game", fake_run_game)
    monkeypatch.setattr(kaggle, "_build_kaggle_batch_run_id", lambda: "kaggle-test")

    result = kaggle.run_config(
        config_path=config_path,
        database_dir=tmp_path / "runs",
        tags=("unit",),
        arcade=fake_arcade,
        trace_output=StringIO(),
    )

    assert [success.game_id for success in result.successes] == ["ls20-aaa"]
    assert [failure.game_id for failure in result.failures] == ["vc33-bbb"]
    assert result.failures[0].message == "boom"
    assert result.failures[0].attempt_count == 1
    assert fake_arcade.closed == ["scorecard-1"]


def test_kaggle_runner_closes_scorecard_after_deadline_clean_stop(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "kaggle.yaml"
    config_path.write_text(
        _minimal_config_text("game_selection: all_available"),
        encoding="utf-8",
    )
    fake_arcade = _FakeArcade()

    def fake_run_game(spec: ParallelGameRunSpec, trace_output):
        del trace_output
        return GameRunResult(
            run_id=spec.run_id,
            game_id=spec.game_id,
            stop_reason="runtime_deadline_reached",
            step_count=0,
        )

    monkeypatch.setattr(kaggle, "_run_kaggle_game", fake_run_game)
    monkeypatch.setattr(kaggle, "_build_kaggle_batch_run_id", lambda: "kaggle-test")

    result = kaggle.run_config(
        config_path=config_path,
        database_dir=tmp_path / "runs",
        tags=("unit",),
        arcade=fake_arcade,
        trace_output=StringIO(),
        deadline_monotonic=123.0,
    )

    assert result.failures == ()
    assert [success.result.stop_reason for success in result.successes] == [
        "runtime_deadline_reached",
        "runtime_deadline_reached",
    ]
    assert fake_arcade.closed == ["scorecard-1"]


def test_kaggle_run_game_passes_deadline_into_runtime_config(
    monkeypatch,
    tmp_path,
) -> None:
    captured_deadlines: list[float | None] = []
    spec = ParallelGameRunSpec(
        game_index=0,
        game_id="ls20-aaa",
        run_id="run-1",
        database_path=tmp_path / "memory.sqlite",
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="ls20-aaa",
            max_actions_per_level=1,
        ),
        arc_environment=object(),
        deadline_monotonic=123.0,
    )

    class FakeRuntimeLoop:
        def __init__(self, orchestrator, *, trace_output=None, live_turn_monitor=None):
            del orchestrator, trace_output, live_turn_monitor

        def run(self, *, config, environment, environment_config):
            del environment, environment_config
            captured_deadlines.append(config.deadline_monotonic)
            return _result(spec)

    monkeypatch.setattr(kaggle, "_build_model_registry", lambda **kwargs: object())
    monkeypatch.setattr(kaggle, "_build_orchestrator", lambda *args, **kwargs: object())
    monkeypatch.setattr(kaggle, "RuntimeLoop", FakeRuntimeLoop)

    result = kaggle._run_kaggle_game(spec, StringIO())

    assert result.run_id == "run-1"
    assert captured_deadlines == [123.0]


def test_kaggle_main_exits_zero_for_worker_failures(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    result = ParallelGameRunResult(
        batch_run_id="kaggle-test",
        failures=(
            ParallelGameRunFailure(
                game_index=1,
                game_id="vc33-bbb",
                run_id="run-1",
                database_path=str(tmp_path / "memory-vc33-bbb.sqlite"),
                exception_type="RuntimeError",
                message="boom",
                attempt_count=1,
            ),
        ),
    )

    def fake_run_config(**kwargs):
        del kwargs
        return result

    monkeypatch.setattr(kaggle, "run_config", fake_run_config)

    kaggle.main(
        [
            "--config",
            str(tmp_path / "kaggle.yaml"),
            "--database-dir",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    assert "leaving notebook exit status at 0" in captured.out
