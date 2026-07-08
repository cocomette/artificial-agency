"""Pull Modal Volume SQLite snapshots for the debug dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess

DEFAULT_MODAL_DATABASE = "/vol/runs/memory.sqlite"
DEFAULT_MODAL_SNAPSHOT = "runs/modal-memory.sqlite"
DEFAULT_MODAL_VOLUME = "face-of-agi-runs"
MODAL_RUN_VOLUME_PATH = "/vol/runs"


class ModalPullError(RuntimeError):
    """Raised when a Modal Volume snapshot cannot be downloaded."""


@dataclass(frozen=True)
class ModalSnapshotConfig:
    """User-editable Modal snapshot settings."""

    volume_name: str
    remote_database: str
    local_snapshot: str


Runner = Callable[..., subprocess.CompletedProcess[str]]


def pull_modal_sqlite_snapshot(
    *,
    volume_name: str,
    remote_path: str,
    local_path: str | Path,
    runner: Runner = subprocess.run,
) -> Path:
    """Download a Modal Volume SQLite file and atomically replace local_path."""

    volume = volume_name.strip()
    if not volume:
        raise ModalPullError("Modal volume name cannot be empty.")
    relative_remote_path = volume_relative_path(remote_path)
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.exists():
        temporary.unlink()

    command = [
        "modal",
        "volume",
        "get",
        volume,
        relative_remote_path,
        str(temporary),
    ]
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        _remove_partial_download(temporary)
        raise ModalPullError(str(exc)) from exc
    if completed.returncode != 0:
        _remove_partial_download(temporary)
        raise ModalPullError(_command_error_message(completed))
    if not temporary.exists():
        raise ModalPullError("Modal download finished without creating a snapshot.")

    temporary.replace(destination)
    return destination


def volume_relative_path(remote_path: str) -> str:
    """Return a path relative to the Modal run Volume mount."""

    normalized = remote_path.strip()
    if not normalized:
        raise ModalPullError("Modal remote database path cannot be empty.")

    if normalized == MODAL_RUN_VOLUME_PATH:
        raise ModalPullError("Modal remote database path must name a SQLite file.")
    prefix = f"{MODAL_RUN_VOLUME_PATH}/"
    if normalized.startswith(prefix):
        normalized = normalized.removeprefix(prefix)
    normalized = normalized.lstrip("/")
    if not normalized:
        raise ModalPullError("Modal remote database path must name a SQLite file.")
    return normalized


def _remove_partial_download(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _command_error_message(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )
    if output:
        return output
    return f"modal volume get failed with exit code {completed.returncode}."
