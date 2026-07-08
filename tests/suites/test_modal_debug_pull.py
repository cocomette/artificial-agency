"""Tests for pulling Modal debug memory snapshots without calling Modal."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from debug.dashboard.modal_snapshot import (
    ModalPullError,
    pull_modal_sqlite_snapshot,
    volume_relative_path,
)


def test_pull_modal_sqlite_snapshot_replaces_target_atomically(tmp_path: Path) -> None:
    target = tmp_path / "memory.sqlite"
    target.write_bytes(b"old")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"new")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    pulled = pull_modal_sqlite_snapshot(
        volume_name="face-of-agi-runs",
        remote_path="/vol/runs/memory.sqlite",
        local_path=target,
        runner=fake_run,
    )

    assert pulled == target
    assert target.read_bytes() == b"new"
    assert commands == [
        [
            "modal",
            "volume",
            "get",
            "face-of-agi-runs",
            "memory.sqlite",
            str(tmp_path / ".memory.sqlite.download"),
        ]
    ]


def test_pull_modal_sqlite_snapshot_removes_failed_download(
    tmp_path: Path,
) -> None:
    target = tmp_path / "memory.sqlite"
    temporary = tmp_path / ".memory.sqlite.download"

    def fake_run(
        command: list[str],
        *,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not found")

    with pytest.raises(ModalPullError, match="not found"):
        pull_modal_sqlite_snapshot(
            volume_name="face-of-agi-runs",
            remote_path="memory.sqlite",
            local_path=target,
            runner=fake_run,
        )

    assert not target.exists()
    assert not temporary.exists()


def test_volume_relative_path_normalizes_modal_run_volume_paths() -> None:
    assert volume_relative_path("/vol/runs/memory.sqlite") == "memory.sqlite"
    assert volume_relative_path("memory.sqlite") == "memory.sqlite"
    assert volume_relative_path("/memory.sqlite") == "memory.sqlite"
    assert volume_relative_path("/vol/runs/nested/memory.sqlite") == (
        "nested/memory.sqlite"
    )

    with pytest.raises(ModalPullError, match="cannot be empty"):
        volume_relative_path("")
    with pytest.raises(ModalPullError, match="must name a SQLite file"):
        volume_relative_path("/vol/runs")
