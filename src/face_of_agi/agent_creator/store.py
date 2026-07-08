"""SQLite store for shared dynamic agent-updater role revisions."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from face_of_agi.agent_creator.contracts import (
    AgentCreatorGameRequest,
    AgentCreatorRun,
    AgentRoleDefinition,
    AgentRoleSnapshot,
    ClaimedAgentCreatorBatch,
)

GENERAL_SYSTEM_PROMPT_KEY = "general_system_prompt"


class AgentCreatorStore:
    """Persistent shared role revisions and creator game-request queue."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        """Open the shared creator database."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        """Create the shared creator tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_creator_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_role_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    publication_status TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    meta_description TEXT NOT NULL,
                    role_instructions TEXT NOT NULL,
                    created_by_run_id INTEGER,
                    guidance_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_creator_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    request_ids_json TEXT NOT NULL,
                    max_tool_calls INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_creator_tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    call_index INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_creator_game_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    memory_database_path TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_agent_creator_pending_game_request
                ON agent_creator_game_requests (run_id, game_id)
                WHERE status = 'pending';
                """
            )

    def seed_defaults(
        self,
        *,
        roles: tuple[AgentRoleDefinition, ...],
        general_system_prompt: str,
    ) -> AgentRoleSnapshot:
        """Create completed default role revisions if the revision store is empty."""

        validate_role_definitions(roles)
        self.write_setting(GENERAL_SYSTEM_PROMPT_KEY, general_system_prompt)
        existing = self.read_latest_complete_role_snapshot()
        if existing is not None:
            return existing
        for role in roles:
            self.stage_role_revision(
                role=role,
                active=True,
                operation="defaults",
                publication_status="complete",
                completed=True,
            )
        snapshot = self.read_latest_complete_role_snapshot()
        if snapshot is None:
            raise RuntimeError("agent creator defaults did not create a role snapshot")
        return snapshot

    def read_latest_complete_role_snapshot(self) -> AgentRoleSnapshot | None:
        """Return the latest completed active role projection."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT revision.*
                FROM agent_role_revisions revision
                JOIN (
                    SELECT role_name, MAX(id) AS latest_id
                    FROM agent_role_revisions
                    WHERE publication_status = 'complete'
                    GROUP BY role_name
                ) latest ON latest.latest_id = revision.id
                WHERE revision.active = 1
                ORDER BY revision.role_name
                """
            ).fetchall()
            projection_id = connection.execute(
                """
                SELECT MAX(id) AS projection_id
                FROM agent_role_revisions
                WHERE publication_status = 'complete'
                """
            ).fetchone()
        if not rows:
            return None
        roles = tuple(
            AgentRoleDefinition(
                role=str(row["role_name"]),
                meta_description=str(row["meta_description"]),
                role_instructions=str(row["role_instructions"]),
            )
            for row in rows
        )
        validate_role_definitions(roles)
        return AgentRoleSnapshot(
            id=(
                int(projection_id["projection_id"])
                if projection_id is not None
                and projection_id["projection_id"] is not None
                else None
            ),
            roles=roles,
            general_system_prompt=self.read_setting(GENERAL_SYSTEM_PROMPT_KEY),
            metadata={"source": "role_revisions"},
            created_at="",
        )

    def write_role_snapshot(
        self,
        *,
        roles: tuple[AgentRoleDefinition, ...],
        general_system_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRoleSnapshot:
        """Compatibility helper that writes completed active role revisions."""

        del metadata
        validate_role_definitions(roles)
        self.write_setting(GENERAL_SYSTEM_PROMPT_KEY, general_system_prompt)
        current = self.read_latest_complete_role_snapshot()
        replacement_names = {role.role for role in roles}
        if current is not None:
            for role in current.roles:
                if role.role not in replacement_names:
                    self.stage_role_revision(
                        role=role,
                        active=False,
                        operation="snapshot_write",
                        publication_status="complete",
                        completed=True,
                    )
        for role in roles:
            self.stage_role_revision(
                role=role,
                active=True,
                operation="snapshot_write",
                publication_status="complete",
                completed=True,
            )
        snapshot = self.read_latest_complete_role_snapshot()
        if snapshot is None:
            raise RuntimeError("agent creator write_role_snapshot created no roles")
        return snapshot

    def create_creator_run(
        self,
        *,
        request_ids: tuple[int, ...],
        max_tool_calls: int,
    ) -> AgentCreatorRun:
        """Create a running creator workflow record."""

        if max_tool_calls < 0:
            raise ValueError("agent creator max_tool_calls must be non-negative")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_creator_runs (
                    status,
                    request_ids_json,
                    max_tool_calls
                )
                VALUES ('running', ?, ?)
                """,
                (json.dumps(list(request_ids)), max_tool_calls),
            )
            row = connection.execute(
                "SELECT * FROM agent_creator_runs WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return _row_to_creator_run(row)

    def complete_creator_run(self, run_id: int) -> None:
        """Publish staged role revisions for a successful creator run."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE agent_role_revisions
                SET publication_status = 'complete',
                    completed_at = CURRENT_TIMESTAMP
                WHERE created_by_run_id = ?
                  AND publication_status = 'staged'
                """,
                (run_id,),
            )
            connection.execute(
                """
                UPDATE agent_creator_runs
                SET status = 'complete',
                    completed_at = CURRENT_TIMESTAMP,
                    error = NULL
                WHERE id = ?
                """,
                (run_id,),
            )
            connection.commit()

    def fail_creator_run(self, run_id: int, error: str) -> None:
        """Fail a creator run and hide all staged revisions from readers."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE agent_role_revisions
                SET publication_status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error = ?
                WHERE created_by_run_id = ?
                  AND publication_status = 'staged'
                """,
                (error, run_id),
            )
            connection.execute(
                """
                UPDATE agent_creator_runs
                SET status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error = ?
                WHERE id = ?
                """,
                (error, run_id),
            )
            connection.commit()

    def record_tool_call(
        self,
        *,
        run_id: int,
        call_index: int,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> int:
        """Record one creator-orchestrator tool call result."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_creator_tool_calls (
                    run_id,
                    call_index,
                    tool_name,
                    arguments_json,
                    status,
                    result_json,
                    error,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    run_id,
                    call_index,
                    tool_name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                    "complete" if ok else "failed",
                    (
                        json.dumps(result, sort_keys=True, ensure_ascii=False)
                        if result is not None
                        else None
                    ),
                    error,
                ),
            )
        return int(cursor.lastrowid)

    def clear_transient_workflow_state(self) -> None:
        """Remove queued/running creator workflow rows while keeping role history."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM agent_creator_tool_calls")
            connection.execute("DELETE FROM agent_creator_runs")
            connection.execute("DELETE FROM agent_creator_game_requests")
            connection.execute(
                """
                DELETE FROM sqlite_sequence
                WHERE name IN (
                    'agent_creator_tool_calls',
                    'agent_creator_runs',
                    'agent_creator_game_requests'
                )
                """
            )
            connection.commit()

    def stage_role_revision(
        self,
        *,
        role: AgentRoleDefinition,
        active: bool,
        operation: str,
        created_by_run_id: int | None = None,
        guidance: dict[str, Any] | None = None,
        publication_status: str = "staged",
        completed: bool = False,
    ) -> int:
        """Write one role revision without exposing staged rows to readers."""

        if active:
            validate_role_definitions((role,))
        elif not role.role.strip():
            raise ValueError("agent role name must not be empty")
        version = self.next_role_version(role.role)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_role_revisions (
                    role_name,
                    version,
                    active,
                    publication_status,
                    operation,
                    meta_description,
                    role_instructions,
                    created_by_run_id,
                    guidance_json,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, """
                + ("CURRENT_TIMESTAMP" if completed else "NULL")
                + """)
                """,
                (
                    role.role,
                    version,
                    1 if active else 0,
                    publication_status,
                    operation,
                    role.meta_description,
                    role.role_instructions,
                    created_by_run_id,
                    json.dumps(guidance or {}, sort_keys=True, ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)

    def next_role_version(self, role_name: str) -> int:
        """Return the next append-only version number for one role name."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(version) AS latest_version
                FROM agent_role_revisions
                WHERE role_name = ?
                """,
                (role_name,),
            ).fetchone()
        latest = row["latest_version"] if row is not None else None
        return int(latest or 0) + 1

    def write_setting(self, key: str, value: str) -> None:
        """Persist one creator setting."""

        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_creator_settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def read_setting(self, key: str) -> str:
        """Read one creator setting, returning an empty string when absent."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM agent_creator_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return "" if row is None else str(row["value"])

    def enqueue_game_request(
        self,
        *,
        run_id: str,
        game_id: str,
        memory_database_path: str,
    ) -> int:
        """Queue one game unless a pending request already exists."""

        if not run_id.strip():
            raise ValueError("agent creator game request run_id must not be empty")
        if not game_id.strip():
            raise ValueError("agent creator game request game_id must not be empty")
        if not memory_database_path.strip():
            raise ValueError(
                "agent creator game request memory_database_path must not be empty"
            )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT id
                FROM agent_creator_game_requests
                WHERE status = 'pending'
                  AND run_id = ?
                  AND game_id = ?
                LIMIT 1
                """,
                (run_id, game_id),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO agent_creator_game_requests (
                    status,
                    run_id,
                    game_id,
                    memory_database_path
                )
                VALUES ('pending', ?, ?, ?)
                """,
                (run_id, game_id, memory_database_path),
            )
            request_id = int(cursor.lastrowid)
            connection.commit()
        return request_id

    def claim_full_batch(
        self,
        *,
        batch_size: int,
    ) -> ClaimedAgentCreatorBatch | None:
        """Atomically claim the oldest full pending game-request batch."""

        if batch_size < 1:
            raise ValueError("agent creator batch_size must be at least 1")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT *
                FROM agent_creator_game_requests
                WHERE status = 'pending'
                ORDER BY id
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
            if len(rows) < batch_size:
                connection.rollback()
                return None
            request_ids = tuple(int(row["id"]) for row in rows)
            connection.executemany(
                """
                UPDATE agent_creator_game_requests
                SET status = 'claimed', claimed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [(request_id,) for request_id in request_ids],
            )
            connection.commit()
        return ClaimedAgentCreatorBatch(
            request_ids=request_ids,
            requests=tuple(_row_to_game_request(row) for row in rows),
        )

    def mark_batch_complete(self, request_ids: tuple[int, ...]) -> None:
        """Mark claimed game requests as completed."""

        self._mark_batch(request_ids, status="completed", error=None)

    def mark_batch_failed(self, request_ids: tuple[int, ...], error: str) -> None:
        """Mark claimed game requests as failed with an explicit error."""

        self._mark_batch(request_ids, status="failed", error=error)

    def _mark_batch(
        self,
        request_ids: tuple[int, ...],
        *,
        status: str,
        error: str | None,
    ) -> None:
        if not request_ids:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                UPDATE agent_creator_game_requests
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error = ?
                WHERE id = ?
                """,
                [(status, error, request_id) for request_id in request_ids],
            )


def validate_role_definitions(roles: tuple[AgentRoleDefinition, ...]) -> None:
    """Require a non-empty role set with unique non-empty role names."""

    if not roles:
        raise ValueError("agent creator role snapshot must contain at least one role")
    seen: set[str] = set()
    for role in roles:
        name = role.role.strip()
        if not name:
            raise ValueError("agent role name must not be empty")
        if name in seen:
            raise ValueError(f"duplicate agent role: {name}")
        if not role.meta_description.strip():
            raise ValueError(f"agent role {name!r} meta_description must not be empty")
        if not role.role_instructions.strip():
            raise ValueError(f"agent role {name!r} role_instructions must not be empty")
        seen.add(name)


def _row_to_creator_run(row: sqlite3.Row) -> AgentCreatorRun:
    request_ids = json.loads(str(row["request_ids_json"]))
    if not isinstance(request_ids, list):
        raise RuntimeError("agent creator run request_ids_json must be an array")
    return AgentCreatorRun(
        id=int(row["id"]),
        status=str(row["status"]),
        request_ids=tuple(int(item) for item in request_ids),
        max_tool_calls=int(row["max_tool_calls"]),
        created_at=str(row["created_at"]),
        completed_at=(
            str(row["completed_at"]) if row["completed_at"] is not None else None
        ),
        error=str(row["error"]) if row["error"] is not None else None,
    )


def _row_to_game_request(row: sqlite3.Row) -> AgentCreatorGameRequest:
    return AgentCreatorGameRequest(
        id=int(row["id"]),
        run_id=str(row["run_id"]),
        game_id=str(row["game_id"]),
        memory_database_path=str(row["memory_database_path"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        claimed_at=(
            str(row["claimed_at"]) if row["claimed_at"] is not None else None
        ),
        completed_at=(
            str(row["completed_at"]) if row["completed_at"] is not None else None
        ),
        error=str(row["error"]) if row["error"] is not None else None,
    )
