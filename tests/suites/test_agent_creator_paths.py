"""Tests for versioned agent creator database path allocation."""

from __future__ import annotations

import sqlite3

from face_of_agi.runtime.agent_creator_paths import (
    allocate_agent_creator_database_path,
    latest_agent_creator_database_path,
)


def test_allocate_agent_creator_database_path_uses_memory_database_directory(
    tmp_path,
) -> None:
    allocation = allocate_agent_creator_database_path(
        "runs/agent_creator",
        memory_database_path=tmp_path / "memory.sqlite",
    )

    assert allocation.path == tmp_path / "agent_creator_01.sqlite"
    assert allocation.copied_from is None


def test_allocate_agent_creator_database_path_increments_existing_numbers(
    tmp_path,
) -> None:
    (tmp_path / "agent_creator_01.sqlite").touch()
    (tmp_path / "agent_creator_02.sqlite").touch()

    allocation = allocate_agent_creator_database_path(
        "runs/agent_creator",
        database_dir=tmp_path,
    )

    assert allocation.path == tmp_path / "agent_creator_03.sqlite"


def test_allocate_agent_creator_database_path_copies_latest_when_requested(
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs"
    data_dir.mkdir()
    source = data_dir / "agent_creator_01.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('kept')")

    allocation = allocate_agent_creator_database_path(
        data_dir / "agent_creator",
        database_dir=run_dir,
        copy_latest=True,
    )

    assert allocation.path == run_dir / "agent_creator_01.sqlite"
    assert allocation.copied_from == source
    with sqlite3.connect(allocation.path) as connection:
        rows = connection.execute("SELECT value FROM marker").fetchall()
    assert rows == [("kept",)]


def test_allocate_agent_creator_database_path_never_seeds_from_run_directory(
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs"
    data_dir.mkdir()
    run_dir.mkdir()
    blessed_source = data_dir / "agent_creator_02.sqlite"
    run_artifact = run_dir / "agent_creator_07.sqlite"
    with sqlite3.connect(blessed_source) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('blessed')")
    with sqlite3.connect(run_artifact) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('run-artifact')")

    allocation = allocate_agent_creator_database_path(
        data_dir / "agent_creator",
        database_dir=run_dir,
        copy_latest=True,
    )

    assert allocation.path == run_dir / "agent_creator_08.sqlite"
    assert allocation.copied_from == blessed_source
    with sqlite3.connect(allocation.path) as connection:
        rows = connection.execute("SELECT value FROM marker").fetchall()
    assert rows == [("blessed",)]


def test_latest_agent_creator_database_path_falls_back_to_unnumbered_file(
    tmp_path,
) -> None:
    unnumbered = tmp_path / "agent_creator.sqlite"
    unnumbered.touch()

    assert latest_agent_creator_database_path(tmp_path) == unnumbered


def test_latest_agent_creator_database_path_uses_literal_base_name(tmp_path) -> None:
    expected = tmp_path / "agent_creator_01_01.sqlite"
    ignored = tmp_path / "agent_creator_02.sqlite"
    expected.touch()
    ignored.touch()

    assert (
        latest_agent_creator_database_path(
            tmp_path,
            base_learned_roles_file=tmp_path / "agent_creator_01",
        )
        == expected
    )
