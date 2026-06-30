"""Tests for local Kaggle workflow helpers."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from face_of_agi.contracts import (
    GameRunResult,
    ParallelGameRunFailure,
    ParallelGameRunResult,
    ParallelGameRunSuccess,
)
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.runtime import kaggle
from face_of_agi.runtime.parallel import ParallelGameRunSpec, ParallelRuntimeLoop


ROOT = Path(__file__).resolve().parents[2]


def _success(index: int = 0) -> ParallelGameRunSuccess:
    game_id = f"game-{index}"
    run_id = f"run-{index}"
    return ParallelGameRunSuccess(
        game_index=index,
        game_id=game_id,
        run_id=run_id,
        database_path=f"memory-{game_id}.sqlite",
        result=GameRunResult(
            run_id=run_id,
            game_id=game_id,
            stop_reason="completed",
        ),
    )


def _failure(index: int = 0) -> ParallelGameRunFailure:
    game_id = f"game-{index}"
    return ParallelGameRunFailure(
        game_index=index,
        game_id=game_id,
        run_id=f"run-{index}",
        database_path=f"memory-{game_id}.sqlite",
        exception_type="RuntimeError",
        message="worker failed",
    )


def _run_main_with_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: ParallelGameRunResult,
) -> list[ParallelGameRunResult]:
    printed: list[ParallelGameRunResult] = []

    def fake_run_config(**_: object) -> ParallelGameRunResult:
        return result

    def fake_print_parallel_result(value: ParallelGameRunResult) -> None:
        printed.append(value)

    monkeypatch.setattr(kaggle, "run_config", fake_run_config)
    monkeypatch.setattr(kaggle, "_print_parallel_result", fake_print_parallel_result)

    kaggle.main(
        [
            "--config",
            "unused.yaml",
            "--database-dir",
            str(tmp_path),
        ]
    )
    return printed


def test_generated_kaggle_artifacts_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".kaggle/" in gitignore
    assert "kaggle/build/" in gitignore
    assert "kaggle/notebooks/submission.ipynb" in gitignore
    assert "kaggle/debug-notebooks/debug.ipynb" in gitignore
    assert "kaggle/debug-notebooks/kernel-metadata.json" in gitignore


def test_kaggle_env_example_documents_owner_and_token_path() -> None:
    example = (ROOT / "kaggle/.env.example").read_text(encoding="utf-8")

    assert "FACE_OF_AGI_KAGGLE_OWNER=" in example
    assert "FACE_OF_AGI_KAGGLE_TOKEN_FILE=.kaggle/access_token" in example


def test_public_kaggle_metadata_templates_are_owner_neutral() -> None:
    metadata_paths = [
        ROOT / "kaggle/notebooks/kernel-metadata.json",
        ROOT / "kaggle/debug-notebooks/kernel-metadata.template.json",
        ROOT / "kaggle/upload/wheelhouse/dataset-metadata.json",
        ROOT / "kaggle/upload/public-games/dataset-metadata.json",
        ROOT / "kaggle/upload/model-dataset/dataset-metadata.json",
    ]

    for path in metadata_paths:
        text = path.read_text(encoding="utf-8")
        assert "kaggle-owner/" in text


def test_sync_metadata_targets_debug_template() -> None:
    script = (ROOT / "kaggle/scripts/sync_kaggle_metadata.py").read_text(
        encoding="utf-8"
    )

    assert 'KAGGLE_ROOT / "debug-notebooks/kernel-metadata.template.json"' in script
    assert 'KAGGLE_ROOT / "debug-notebooks/kernel-metadata.json"' not in script


def test_resolve_kaggle_kernel_ref_accepts_metadata_and_urls(tmp_path) -> None:
    metadata_path = tmp_path / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps({"id": "owner/from-metadata"}),
        encoding="utf-8",
    )
    script = ROOT / "kaggle/scripts/resolve_kaggle_kernel_ref.py"

    from_metadata = subprocess.run(
        [sys.executable, str(script), "", str(metadata_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    from_url = subprocess.run(
        [
            sys.executable,
            str(script),
            "https://www.kaggle.com/code/other-owner/debug-run",
            str(metadata_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert from_metadata.stdout.strip() == "owner/from-metadata"
    assert from_url.stdout.strip() == "other-owner/debug-run"


def test_kaggle_main_keeps_zero_exit_status_for_all_successes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ParallelGameRunResult(
        batch_run_id="batch",
        successes=(_success(),),
    )

    printed = _run_main_with_result(monkeypatch, tmp_path, result)

    assert printed == [result]
    assert "failed games after retries" not in capsys.readouterr().out


def test_kaggle_main_keeps_zero_exit_status_for_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ParallelGameRunResult(
        batch_run_id="batch",
        successes=(_success(0),),
        failures=(_failure(1),),
    )

    printed = _run_main_with_result(monkeypatch, tmp_path, result)

    assert printed == [result]
    output = capsys.readouterr().out
    assert "kaggle completed with failed games after retries" in output
    assert "leaving notebook exit status at 0" in output


def test_kaggle_main_keeps_zero_exit_status_when_all_games_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ParallelGameRunResult(
        batch_run_id="batch",
        failures=(_failure(),),
    )

    printed = _run_main_with_result(monkeypatch, tmp_path, result)

    assert printed == [result]
    output = capsys.readouterr().out
    assert "kaggle completed with failed games after retries" in output
    assert "leaving notebook exit status at 0" in output


def test_parallel_runtime_reports_worker_exception_as_failure(
    tmp_path: Path,
) -> None:
    def run_game(spec: ParallelGameRunSpec, output: object) -> GameRunResult:
        del spec, output
        raise RuntimeError("worker exploded")

    spec = ParallelGameRunSpec(
        game_index=0,
        game_id="game-0",
        run_id="run-0",
        database_path=tmp_path / "memory-game-0.sqlite",
        environment_config=EnvironmentConfig(
            game_index=0,
            game_id="game-0",
            max_actions_per_level=1,
        ),
    )

    result = ParallelRuntimeLoop(
        run_game,
        trace_output=io.StringIO(),
    ).run(
        batch_run_id="batch",
        specs=(spec,),
        max_game_retries=1,
    )

    assert result.successes == ()
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.attempt_count == 2
    assert failure.exception_type == "RuntimeError"
    assert failure.message == "worker exploded"
