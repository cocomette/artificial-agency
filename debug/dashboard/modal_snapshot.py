"""Pull Modal Volume SQLite snapshots for the debug dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess

DEFAULT_MODAL_DATABASE = "memory.sqlite"
DEFAULT_MODAL_DATABASE_PATTERN = r"^memory-game-index-.*\.sqlite$"
DEFAULT_MODAL_RUN_FOLDER = ""
DEFAULT_MODAL_SNAPSHOT = "runs/memory.sqlite"
DEFAULT_MODAL_VOLUME = "face-of-agi-runs"
MODAL_RUN_VOLUME_PATH = "/vol/runs"


class ModalPullError(RuntimeError):
    """Raised when a Modal Volume snapshot cannot be downloaded."""


@dataclass(frozen=True)
class ModalSnapshotConfig:
    """User-editable Modal snapshot settings."""

    volume_name: str
    run_folder: str
    database_name: str
    database_pattern: str
    local_snapshot: str

    @property
    def remote_database(self) -> str:
        """Return the selected SQLite path under the Modal run Volume mount."""

        return remote_database_path(
            run_folder=self.run_folder,
            database_name=self.database_name,
        )

    @property
    def local_database(self) -> str:
        """Return the local selected SQLite snapshot path."""

        return str(self.local_database_path(self.database_name))

    def local_database_path(self, database_name: str) -> Path:
        """Return the local batch snapshot path for one database file."""

        name = Path(volume_relative_path(database_name)).name
        base = Path(self.local_snapshot).parent
        return base / name


Runner = Callable[..., subprocess.CompletedProcess[str]]


def remote_database_path(*, run_folder: str, database_name: str) -> str:
    """Return a Modal run-Volume path for one database in a run folder."""

    relative = volume_relative_path(database_name)
    folder = run_folder.strip().strip("/")
    if folder:
        return f"{MODAL_RUN_VOLUME_PATH}/{folder}/{relative}"
    return f"{MODAL_RUN_VOLUME_PATH}/{relative}"


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


def pull_modal_sqlite_snapshots(
    *,
    volume_name: str,
    run_folder: str,
    pattern: str = DEFAULT_MODAL_DATABASE_PATTERN,
    local_folder: str | Path,
    runner: Runner = subprocess.run,
) -> dict[str, Path]:
    """Download all matching SQLite files from one Modal run folder."""

    databases = list_modal_sqlite_databases(
        volume_name=volume_name,
        run_folder=run_folder,
        pattern=pattern,
        runner=runner,
    )
    if not databases:
        raise ModalPullError(f"No Modal SQLite databases match /{pattern}/.")

    destination_folder = Path(local_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    pulled: dict[str, Path] = {}
    for database in databases:
        destination = destination_folder / Path(database).name
        pulled[database] = pull_modal_sqlite_snapshot(
            volume_name=volume_name,
            remote_path=remote_database_path(
                run_folder=run_folder,
                database_name=database,
            ),
            local_path=destination,
            runner=runner,
        )
    return pulled


def list_modal_run_folders(
    *,
    volume_name: str,
    runner: Runner = subprocess.run,
) -> list[str]:
    """Return top-level run folders in the Modal run Volume."""

    volume = volume_name.strip()
    if not volume:
        raise ModalPullError("Modal volume name cannot be empty.")
    command = [
        "modal",
        "volume",
        "ls",
        "--json",
        volume,
    ]
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ModalPullError(str(exc)) from exc
    if completed.returncode != 0:
        raise ModalPullError(_command_error_message(completed))

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ModalPullError("Modal volume ls returned invalid JSON.") from exc
    return _run_folders_from_volume_ls_json(payload)


def list_modal_sqlite_databases(
    *,
    volume_name: str,
    run_folder: str,
    pattern: str = DEFAULT_MODAL_DATABASE_PATTERN,
    runner: Runner = subprocess.run,
) -> list[str]:
    """Return SQLite files in one Modal run folder matching a regex pattern."""

    volume = volume_name.strip()
    if not volume:
        raise ModalPullError("Modal volume name cannot be empty.")
    folder = volume_relative_path(run_folder) if run_folder.strip() else ""
    database_pattern = pattern.strip() or DEFAULT_MODAL_DATABASE_PATTERN
    try:
        compiled_pattern = re.compile(database_pattern)
    except re.error as exc:
        raise ModalPullError(f"invalid Modal database regex: {exc}") from exc
    command = [
        "modal",
        "volume",
        "ls",
        "--json",
        volume,
    ]
    if folder:
        command.append(folder)
    try:
        completed = runner(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ModalPullError(str(exc)) from exc
    if completed.returncode != 0:
        raise ModalPullError(_command_error_message(completed))

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ModalPullError("Modal volume ls returned invalid JSON.") from exc
    return _sqlite_files_from_volume_ls_json(payload, pattern=compiled_pattern)


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


def _run_folders_from_volume_ls_json(payload: object) -> list[str]:
    """Extract directory names from Modal CLI JSON output variants."""

    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("items") or payload.get("data")
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ModalPullError("Modal volume ls JSON must contain a list.")

    folders: list[str] = []
    for entry in entries:
        name: str | None = None
        is_directory = False
        if isinstance(entry, str):
            name = entry.rstrip("/")
            is_directory = entry.endswith("/")
        elif isinstance(entry, dict):
            raw_name = entry.get("name") or entry.get("path") or entry.get("filename")
            if isinstance(raw_name, str):
                name = raw_name.rstrip("/")
            raw_type = str(entry.get("type") or entry.get("kind") or "").lower()
            is_directory = (
                raw_type in {"dir", "directory", "folder"}
                or bool(entry.get("is_dir") or entry.get("is_directory"))
                or (isinstance(raw_name, str) and raw_name.endswith("/"))
            )
        if not name or "/" in name or not is_directory:
            continue
        folders.append(name)
    return sorted(set(folders), reverse=True)


def _sqlite_files_from_volume_ls_json(
    payload: object,
    *,
    pattern: re.Pattern[str],
) -> list[str]:
    """Extract matching SQLite file names from Modal CLI JSON output variants."""

    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("items") or payload.get("data")
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ModalPullError("Modal volume ls JSON must contain a list.")

    files: list[str] = []
    for entry in entries:
        name: str | None = None
        is_directory = False
        if isinstance(entry, str):
            name = entry.rstrip("/")
            is_directory = entry.endswith("/")
        elif isinstance(entry, dict):
            raw_name = entry.get("name") or entry.get("path") or entry.get("filename")
            if isinstance(raw_name, str):
                name = raw_name.rstrip("/")
            raw_type = str(entry.get("type") or entry.get("kind") or "").lower()
            is_directory = (
                raw_type in {"dir", "directory", "folder"}
                or bool(entry.get("is_dir") or entry.get("is_directory"))
                or (isinstance(raw_name, str) and raw_name.endswith("/"))
            )
        if not name or is_directory:
            continue
        filename = Path(name).name
        if pattern.fullmatch(filename):
            files.append(filename)
    return sorted(set(files))


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
