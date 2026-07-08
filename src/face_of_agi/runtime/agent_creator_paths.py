"""Path helpers for versioned agent creator SQLite databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3


@dataclass(frozen=True, slots=True)
class AgentCreatorDatabaseAllocation:
    """Resolved writable agent creator database path for one runtime run."""

    path: Path
    copied_from: Path | None = None


@dataclass(frozen=True, slots=True)
class _NumberedDatabase:
    path: Path
    number: int
    width: int


def allocate_agent_creator_database_path(
    base_learned_roles_file: str | Path,
    *,
    memory_database_path: str | Path | None = None,
    database_dir: str | Path | None = None,
    copy_latest: bool = False,
) -> AgentCreatorDatabaseAllocation:
    """Return the next run-local database path for a learned-role family."""

    configured = Path(base_learned_roles_file)
    directory = _effective_directory(
        configured,
        memory_database_path=memory_database_path,
        database_dir=database_dir,
    )
    stem = _base_stem(configured)
    suffix = configured.suffix or ".sqlite"
    numbered_latest = _latest_numbered_database(
        directory,
        stem=stem,
        suffix=suffix,
    )
    next_number = (numbered_latest.number + 1) if numbered_latest else 1
    width = max(2, len(str(next_number)))
    if numbered_latest is not None:
        width = max(width, numbered_latest.width)

    path = directory / f"{stem}_{next_number:0{width}d}{suffix}"
    source_latest = latest_agent_creator_database_path(
        _configured_directory(configured),
        base_learned_roles_file=configured,
    )
    copied_from = source_latest if copy_latest and source_latest is not None else None
    if copied_from is not None:
        _copy_sqlite_database(copied_from, path)
    return AgentCreatorDatabaseAllocation(path=path, copied_from=copied_from)


def latest_agent_creator_database_path(
    directory: str | Path,
    *,
    base_learned_roles_file: str | Path = "agent_creator",
) -> Path | None:
    """Return the latest numbered database in a learned-role family."""

    configured = Path(base_learned_roles_file)
    search_dir = Path(directory)
    stem = _base_stem(configured)
    suffix = configured.suffix or ".sqlite"
    latest = _latest_numbered_database(
        search_dir,
        stem=stem,
        suffix=suffix,
    )
    if latest is not None:
        return latest.path
    unnumbered = search_dir / f"{stem}{suffix}"
    return unnumbered if unnumbered.exists() else None


def _effective_directory(
    configured: Path,
    *,
    memory_database_path: str | Path | None,
    database_dir: str | Path | None,
) -> Path:
    if database_dir is not None:
        return Path(database_dir)
    if memory_database_path is not None:
        memory_path = Path(memory_database_path)
        if _looks_like_directory(memory_path):
            return memory_path
        return memory_path.parent
    return configured.parent if configured.parent != Path(".") else Path("data")


def _configured_directory(configured: Path) -> Path:
    return configured.parent if configured.parent != Path(".") else Path("data")


def _looks_like_directory(path: Path) -> bool:
    return path.suffix == "" or (path.exists() and path.is_dir())


def _base_stem(configured: Path) -> str:
    return configured.stem or configured.name or "agent_creator"


def _latest_numbered_database(
    directory: Path,
    *,
    stem: str,
    suffix: str,
) -> _NumberedDatabase | None:
    if not directory.is_dir():
        return None
    pattern = re.compile(
        rf"^{re.escape(stem)}_(?P<number>\d+){re.escape(suffix)}$"
    )
    latest: _NumberedDatabase | None = None
    for candidate in directory.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match is None:
            continue
        number_text = match.group("number")
        numbered = _NumberedDatabase(
            path=candidate,
            number=int(number_text),
            width=len(number_text),
        )
        if latest is None or numbered.number > latest.number:
            latest = numbered
    return latest


def _copy_sqlite_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _reset_sqlite_files(target)
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _reset_sqlite_files(path: Path) -> None:
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        if candidate.exists():
            candidate.unlink()
