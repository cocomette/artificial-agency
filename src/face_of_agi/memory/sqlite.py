"""SQLite primitives for the framework memory domains."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

from face_of_agi.contracts import (
    EExperimentRecord,
    MStateRecord,
    MemoryDomain,
    MemoryRecord,
    ObservationRef,
    RoleContext,
)
from face_of_agi.frames import from_memory_jsonable, to_memory_jsonable

TableName = Literal["state_records", "experimental_records"]

_TABLE_DOMAINS: dict[TableName, MemoryDomain] = {
    "state_records": "state",
    "experimental_records": "experimental",
}


class SQLiteDatabase:
    """Small SQLite wrapper for M states and temporary memory records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        """Open a connection with row access enabled."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        """Create the generic memory tables and dedicated M state table."""

        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS state_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    step INTEGER,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS experimental_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    step INTEGER,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS m_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    step INTEGER,
                    frame_index INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    current_observation_json TEXT NOT NULL,
                    chosen_action_json TEXT NOT NULL,
                    world_context_json TEXT NOT NULL,
                    goal_context_json TEXT NOT NULL,
                    agent_context_json TEXT NOT NULL,
                    agent_trace_json TEXT NOT NULL,
                    world_prediction_json TEXT,
                    goal_prediction_json TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS e_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    source_observation_ref_json TEXT NOT NULL,
                    tool_call_json TEXT NOT NULL,
                    output_observation_json TEXT NOT NULL,
                    tool_result_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_column(
                connection,
                table="m_states",
                column="world_prediction_json",
                definition="TEXT",
            )
            self._ensure_column(
                connection,
                table="m_states",
                column="goal_prediction_json",
                definition="TEXT",
            )

    def write_record(
        self,
        table: TableName,
        *,
        run_id: str,
        game_id: str,
        kind: str,
        payload: Any,
        step: int | None = None,
    ) -> MemoryRecord:
        """Write one generic memory record and return its stored form."""

        self._validate_table(table)
        payload_json = _to_json(payload)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                INSERT INTO {table} (run_id, game_id, step, kind, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, game_id, step, kind, payload_json),
            )
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                f"SELECT * FROM {table} WHERE id = ?",
                (record_id,),
            ).fetchone()

        return self._row_to_record(table, row)

    def list_records(
        self,
        table: TableName,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[MemoryRecord]:
        """List generic records, optionally scoped to a run or game."""

        self._validate_table(table)
        clauses: list[str] = []
        values: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} {where} ORDER BY id",
                values,
            ).fetchall()

        return [self._row_to_record(table, row) for row in rows]

    def write_m_state(
        self,
        *,
        game_id: str,
        run_id: str,
        step: int | None,
        frame_index: int,
        frame_count: int,
        current_observation: Any,
        chosen_action: Any,
        world_context: RoleContext,
        goal_context: RoleContext,
        agent_context: RoleContext,
        agent_trace: Any,
        world_prediction: Any | None = None,
        goal_prediction: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Write one complete M state row for a frame turn."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO m_states (
                    game_id,
                    run_id,
                    step,
                    frame_index,
                    frame_count,
                    current_observation_json,
                    chosen_action_json,
                    world_context_json,
                    goal_context_json,
                    agent_context_json,
                    agent_trace_json,
                    world_prediction_json,
                    goal_prediction_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    run_id,
                    step,
                    frame_index,
                    frame_count,
                    _to_json(current_observation),
                    _to_json(chosen_action),
                    _to_json(world_context),
                    _to_json(goal_context),
                    _to_json(agent_context),
                    _to_json(agent_trace),
                    _to_nullable_json(world_prediction),
                    _to_nullable_json(goal_prediction),
                    _to_json(metadata or {}),
                ),
            )
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (record_id,),
            ).fetchone()

        return self._row_to_m_state(row)

    def read_latest_m_state(self, *, game_id: str) -> MStateRecord | None:
        """Return the newest M state row for a game, if any."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM m_states
                WHERE game_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_m_state(row)

    def list_m_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List M state rows, optionally scoped to one game."""

        values: list[str] = []
        where = ""
        if game_id is not None:
            where = "WHERE game_id = ?"
            values.append(game_id)

        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM m_states {where} ORDER BY id",
                values,
            ).fetchall()

        return [self._row_to_m_state(row) for row in rows]

    def cleanup_m_states_keep_latest_per_game(self) -> None:
        """Keep only the newest M state row for each game."""

        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM m_states
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM m_states
                    GROUP BY game_id
                )
                """
            )

    def clear_m_states(self) -> None:
        """Delete all dedicated M state rows."""

        with self.connect() as connection:
            connection.execute("DELETE FROM m_states")

    def write_e_experiment(
        self,
        *,
        game_id: str,
        run_id: str,
        turn_id: int,
        tool_name: str,
        source_observation_ref: Any,
        tool_call: Any,
        output_observation: Any,
        tool_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EExperimentRecord:
        """Write one experimental tool output row."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO e_experiments (
                    game_id,
                    run_id,
                    turn_id,
                    tool_name,
                    source_observation_ref_json,
                    tool_call_json,
                    output_observation_json,
                    tool_result_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    run_id,
                    turn_id,
                    tool_name,
                    _to_json(source_observation_ref),
                    _to_json(tool_call),
                    _to_json(output_observation),
                    _to_json(tool_result),
                    _to_json(metadata or {}),
                ),
            )
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM e_experiments WHERE id = ?",
                (record_id,),
            ).fetchone()

        return self._row_to_e_experiment(row)

    def read_e_experiment(self, *, ref_id: str | int) -> EExperimentRecord | None:
        """Return one E experiment row by its reference id."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM e_experiments WHERE id = ?",
                (str(ref_id),),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_e_experiment(row)

    def list_e_experiments(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[EExperimentRecord]:
        """List E experiment rows, optionally scoped to a run or game."""

        clauses: list[str] = []
        values: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM e_experiments {where} ORDER BY id",
                values,
            ).fetchall()

        return [self._row_to_e_experiment(row) for row in rows]

    def cleanup_e_experiments_keep_latest_turns_per_game(
        self,
        *,
        run_id: str,
        max_turns: int,
        game_id: str | None = None,
    ) -> None:
        """Keep only the latest distinct E turns per game for one run."""

        if max_turns < 1:
            raise ValueError("experimental memory turn buffer must be at least 1")

        if game_id is None:
            with self.connect() as connection:
                game_ids = [
                    str(row["game_id"])
                    for row in connection.execute(
                        """
                        SELECT DISTINCT game_id
                        FROM e_experiments
                        WHERE run_id = ?
                        """,
                        (run_id,),
                    ).fetchall()
                ]
            for stored_game_id in game_ids:
                self.cleanup_e_experiments_keep_latest_turns_per_game(
                    run_id=run_id,
                    game_id=stored_game_id,
                    max_turns=max_turns,
                )
            return

        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM e_experiments
                WHERE run_id = ?
                  AND game_id = ?
                  AND turn_id NOT IN (
                    SELECT turn_id
                    FROM (
                        SELECT DISTINCT turn_id
                        FROM e_experiments
                        WHERE run_id = ?
                          AND game_id = ?
                        ORDER BY turn_id DESC
                        LIMIT ?
                    )
                  )
                """,
                (run_id, game_id, run_id, game_id, max_turns),
            )

    def clear_e_experiments(self) -> None:
        """Delete all dedicated E experiment rows."""

        with self.connect() as connection:
            connection.execute("DELETE FROM e_experiments")

    def clear_memory_tables(self) -> None:
        """Delete all rows from current and legacy memory tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                DELETE FROM m_states;
                DELETE FROM e_experiments;
                DELETE FROM state_records;
                DELETE FROM experimental_records;
                """
            )

    def _row_to_record(self, table: TableName, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            domain=_TABLE_DOMAINS[table],
            run_id=str(row["run_id"]),
            game_id=str(row["game_id"]),
            step=row["step"],
            kind=str(row["kind"]),
            payload=from_memory_jsonable(json.loads(str(row["payload_json"]))),
            created_at=str(row["created_at"]),
        )

    def _row_to_m_state(self, row: sqlite3.Row) -> MStateRecord:
        return MStateRecord(
            id=int(row["id"]),
            game_id=str(row["game_id"]),
            run_id=str(row["run_id"]),
            step=row["step"],
            frame_index=int(row["frame_index"]),
            frame_count=int(row["frame_count"]),
            current_observation=from_memory_jsonable(
                json.loads(str(row["current_observation_json"]))
            ),
            chosen_action=from_memory_jsonable(
                json.loads(str(row["chosen_action_json"]))
            ),
            world_context=_role_context_from_json(row["world_context_json"]),
            goal_context=_role_context_from_json(row["goal_context_json"]),
            agent_context=_role_context_from_json(row["agent_context_json"]),
            agent_trace=from_memory_jsonable(
                json.loads(str(row["agent_trace_json"]))
            ),
            world_prediction=_from_nullable_json(row["world_prediction_json"]),
            goal_prediction=_from_nullable_json(row["goal_prediction_json"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
        )

    def _row_to_e_experiment(self, row: sqlite3.Row) -> EExperimentRecord:
        return EExperimentRecord(
            id=int(row["id"]),
            game_id=str(row["game_id"]),
            run_id=str(row["run_id"]),
            turn_id=int(row["turn_id"]),
            tool_name=str(row["tool_name"]),
            source_observation_ref=_observation_ref_from_json(
                row["source_observation_ref_json"]
            ),
            tool_call=from_memory_jsonable(json.loads(str(row["tool_call_json"]))),
            output_observation=from_memory_jsonable(
                json.loads(str(row["output_observation_json"]))
            ),
            tool_result=from_memory_jsonable(json.loads(str(row["tool_result_json"]))),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
        )

    def _validate_table(self, table: TableName) -> None:
        if table not in _TABLE_DOMAINS:
            raise ValueError(f"unknown memory table: {table}")

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _to_json(value: Any) -> str:
    """Serialize framework objects to stable JSON for SQLite storage."""

    return json.dumps(to_memory_jsonable(value), sort_keys=True)


def _to_nullable_json(value: Any | None) -> str | None:
    """Serialize an optional framework object to stable JSON."""

    if value is None:
        return None
    return _to_json(value)


def _from_nullable_json(value: Any) -> dict[str, Any] | None:
    """Deserialize an optional JSON object stored in SQLite."""

    if value is None:
        return None
    return from_memory_jsonable(json.loads(str(value)))


def _role_context_from_json(value: Any) -> RoleContext:
    """Deserialize one stored role context."""

    loaded = json.loads(str(value))
    if not isinstance(loaded, dict):
        return RoleContext()
    return RoleContext(
        general=str(loaded.get("general", "")),
        game=str(loaded.get("game", "")),
    )


def _observation_ref_from_json(value: Any) -> ObservationRef:
    """Deserialize one stored observation reference."""

    loaded = json.loads(str(value))
    if not isinstance(loaded, dict):
        raise ValueError("stored observation reference must be a JSON object")
    return ObservationRef(
        memory=loaded.get("memory", "state"),
        id=str(loaded.get("id", "")),
    )
