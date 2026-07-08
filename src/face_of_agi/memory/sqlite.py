"""SQLite primitives for online-learner runtime memory."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from face_of_agi.contracts import MStateRecord, RunMetadataRecord, TurnMetrics
from face_of_agi.debug.contracts import ModelInputDebugRecord
from face_of_agi.frames import from_memory_jsonable, to_memory_jsonable
from face_of_agi.runtime import timing as runtime_timing

_CURRENT_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "m_states": (
        "id",
        "game_id",
        "run_id",
        "step",
        "frame_index",
        "frame_count",
        "current_observation_json",
        "chosen_action_json",
        "learner_snapshot_json",
        "learner_trace_json",
        "turn_metrics_json",
        "metadata_json",
        "created_at",
    ),
    "learner_artifacts": (
        "id",
        "game_id",
        "run_id",
        "turn_id",
        "kind",
        "payload_json",
        "metadata_json",
        "created_at",
    ),
    "model_input_debug_records": (
        "id",
        "m_state_id",
        "run_id",
        "game_id",
        "turn_id",
        "call_slot",
        "provider",
        "model",
        "phase",
        "attempt",
        "request_json",
        "usage_json",
        "metadata_json",
        "created_at",
    ),
    "run_metadata": (
        "id",
        "game_id",
        "run_id",
        "kind",
        "metadata_json",
        "created_at",
    ),
}


class SQLiteDatabase:
    """Small SQLite wrapper for committed turns and learner artifacts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        """Open a connection with row access enabled."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        """Create the online-learner memory tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS state_records;
                DROP TABLE IF EXISTS experimental_records;
                DROP TABLE IF EXISTS e_experiments;

                CREATE TABLE IF NOT EXISTS m_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    step INTEGER,
                    frame_index INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    current_observation_json TEXT NOT NULL,
                    chosen_action_json TEXT,
                    learner_snapshot_json TEXT NOT NULL,
                    learner_trace_json TEXT,
                    turn_metrics_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS learner_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS model_input_debug_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    m_state_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    call_slot TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    phase TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    request_json TEXT NOT NULL,
                    usage_json TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS run_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._require_current_schema(connection)

    def write_run_metadata(
        self,
        *,
        game_id: str,
        run_id: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunMetadataRecord:
        """Write one run-level metadata row."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_metadata (game_id, run_id, kind, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (game_id, run_id, kind, _to_json(metadata or {})),
            )
            row = connection.execute(
                "SELECT * FROM run_metadata WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return self._row_to_run_metadata(row)

    def list_run_metadata(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
        kind: str | None = None,
    ) -> list[RunMetadataRecord]:
        """List run-level metadata rows, optionally filtered."""

        clauses: list[str] = []
        values: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM run_metadata {where} ORDER BY id",
                values,
            ).fetchall()
        return [self._row_to_run_metadata(row) for row in rows]

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
        learner_snapshot: Any,
        learner_trace: Any,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Write one complete online-learner turn row."""

        with runtime_timing.span("sqlite.write_m_state.execute"):
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
                        learner_snapshot_json,
                        learner_trace_json,
                        turn_metrics_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id,
                        run_id,
                        step,
                        frame_index,
                        frame_count,
                        _to_json(current_observation),
                        _to_json(chosen_action),
                        _to_json(learner_snapshot),
                        _to_json(learner_trace),
                        _to_json(turn_metrics or TurnMetrics()),
                        _to_json(metadata or {}),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM m_states WHERE id = ?",
                    (int(cursor.lastrowid),),
                ).fetchone()
        return self._row_to_m_state(row)

    def prewrite_m_state(
        self,
        *,
        game_id: str,
        run_id: str,
        step: int | None,
        frame_index: int,
        frame_count: int,
        current_observation: Any,
        learner_snapshot: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Write the source row for a frame before the learner acts."""

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
                    learner_snapshot_json,
                    learner_trace_json,
                    turn_metrics_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)
                """,
                (
                    game_id,
                    run_id,
                    step,
                    frame_index,
                    frame_count,
                    _to_json(current_observation),
                    _to_json(learner_snapshot),
                    _to_json(TurnMetrics()),
                    _to_json(metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return self._row_to_m_state(row)

    def complete_m_state(
        self,
        *,
        state_id: int,
        chosen_action: Any,
        learner_snapshot: Any,
        learner_trace: Any,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Complete a prewritten online-learner turn row."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE m_states
                SET
                    chosen_action_json = ?,
                    learner_snapshot_json = ?,
                    learner_trace_json = ?,
                    turn_metrics_json = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    _to_json(chosen_action),
                    _to_json(learner_snapshot),
                    _to_json(learner_trace),
                    _to_json(turn_metrics or TurnMetrics()),
                    _to_json(metadata or {}),
                    state_id,
                ),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(f"unknown M state row: {state_id}")
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (state_id,),
            ).fetchone()
        return self._row_to_m_state(row)

    def read_latest_m_state(self, *, game_id: str) -> MStateRecord | None:
        """Return the newest complete M state row for a game, if any."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM m_states
                WHERE game_id = ?
                  AND chosen_action_json IS NOT NULL
                  AND learner_trace_json IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()
        return self._row_to_m_state(row) if row is not None else None

    def read_m_state_source(self, *, state_id: int) -> MStateRecord | None:
        """Read one M row by id, including incomplete source rows."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (state_id,),
            ).fetchone()
        return self._row_to_m_state(row) if row is not None else None

    def list_m_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List complete online-learner M state rows."""

        clauses = [
            "chosen_action_json IS NOT NULL",
            "learner_trace_json IS NOT NULL",
        ]
        values: list[str] = []
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)
        where = f"WHERE {' AND '.join(clauses)}"
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
            connection.execute(
                """
                DELETE FROM model_input_debug_records
                WHERE m_state_id NOT IN (SELECT id FROM m_states)
                """
            )

    def clear_m_states(self) -> None:
        """Delete all dedicated M state rows."""

        with self.connect() as connection:
            connection.execute("DELETE FROM model_input_debug_records")
            connection.execute("DELETE FROM m_states")

    def write_learner_artifact(
        self,
        *,
        game_id: str,
        run_id: str,
        turn_id: int,
        kind: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one learner debug artifact row."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO learner_artifacts (
                    game_id,
                    run_id,
                    turn_id,
                    kind,
                    payload_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    run_id,
                    turn_id,
                    kind,
                    _to_json(payload),
                    _to_json(metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM learner_artifacts WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return _row_to_learner_artifact(row)

    def list_learner_artifacts(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List learner artifact rows."""

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
                f"SELECT * FROM learner_artifacts {where} ORDER BY id",
                values,
            ).fetchall()
        return [_row_to_learner_artifact(row) for row in rows]

    def clear_learner_artifacts(self) -> None:
        """Delete all learner artifact rows."""

        with self.connect() as connection:
            connection.execute("DELETE FROM learner_artifacts")

    def write_model_input_debug_record(
        self,
        *,
        m_state_id: int,
        run_id: str,
        game_id: str,
        turn_id: int,
        call_slot: str,
        provider: str,
        model: str | None,
        phase: str,
        attempt: int,
        request: dict[str, Any],
        usage: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelInputDebugRecord:
        """Write one debug request record for passive inspection."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_input_debug_records (
                    m_state_id,
                    run_id,
                    game_id,
                    turn_id,
                    call_slot,
                    provider,
                    model,
                    phase,
                    attempt,
                    request_json,
                    usage_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m_state_id,
                    run_id,
                    game_id,
                    turn_id,
                    call_slot,
                    provider,
                    model,
                    phase,
                    attempt,
                    _to_json(request),
                    _to_nullable_json(usage),
                    _to_json(metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM model_input_debug_records WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return self._row_to_model_input_debug_record(row)

    def list_model_input_debug_records(
        self,
        *,
        m_state_id: int | None = None,
        run_id: str | None = None,
        game_id: str | None = None,
        turn_id: int | None = None,
    ) -> list[ModelInputDebugRecord]:
        """List debug request records."""

        clauses: list[str] = []
        values: list[Any] = []
        if m_state_id is not None:
            clauses.append("m_state_id = ?")
            values.append(m_state_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if game_id is not None:
            clauses.append("game_id = ?")
            values.append(game_id)
        if turn_id is not None:
            clauses.append("turn_id = ?")
            values.append(turn_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM model_input_debug_records {where} ORDER BY id",
                values,
            ).fetchall()
        return [self._row_to_model_input_debug_record(row) for row in rows]

    def clear_memory_tables(self) -> None:
        """Delete all rows from current memory tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                DELETE FROM model_input_debug_records;
                DELETE FROM learner_artifacts;
                DELETE FROM m_states;
                DELETE FROM run_metadata;
                """
            )

    def _row_to_run_metadata(self, row: sqlite3.Row) -> RunMetadataRecord:
        return RunMetadataRecord(
            id=int(row["id"]),
            game_id=str(row["game_id"]),
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
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
            chosen_action=_from_nullable_json(row["chosen_action_json"]),
            learner_snapshot=from_memory_jsonable(
                json.loads(str(row["learner_snapshot_json"]))
            ),
            learner_trace=_from_nullable_json(row["learner_trace_json"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
            turn_metrics=_turn_metrics_from_json(row["turn_metrics_json"]),
        )

    def _row_to_model_input_debug_record(
        self,
        row: sqlite3.Row,
    ) -> ModelInputDebugRecord:
        return ModelInputDebugRecord(
            id=int(row["id"]),
            m_state_id=int(row["m_state_id"]),
            run_id=str(row["run_id"]),
            game_id=str(row["game_id"]),
            turn_id=int(row["turn_id"]),
            call_slot=str(row["call_slot"]),
            provider=str(row["provider"]),
            model=(str(row["model"]) if row["model"] is not None else None),
            phase=str(row["phase"]),
            attempt=int(row["attempt"]),
            request=from_memory_jsonable(json.loads(str(row["request_json"]))),
            usage=_from_nullable_json(row["usage_json"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
        )

    def _require_current_schema(self, connection: sqlite3.Connection) -> None:
        """Fail fast when a local DB file predates the current schema."""

        for table, expected in _CURRENT_TABLE_COLUMNS.items():
            found = tuple(
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if found == expected:
                continue

            missing = sorted(set(expected) - set(found))
            extra = sorted(set(found) - set(expected))
            details: list[str] = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if extra:
                details.append(f"extra columns: {', '.join(extra)}")
            if not details:
                details.append("column order differs from the current schema")
            raise RuntimeError(
                f"SQLite table {table!r} in {self.path} does not match the "
                "current memory schema; reset this disposable local database "
                "before running again. "
                + "; ".join(details)
            )


def _row_to_learner_artifact(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "game_id": str(row["game_id"]),
        "run_id": str(row["run_id"]),
        "turn_id": int(row["turn_id"]),
        "kind": str(row["kind"]),
        "payload": from_memory_jsonable(json.loads(str(row["payload_json"]))),
        "metadata": from_memory_jsonable(json.loads(str(row["metadata_json"]))),
        "created_at": str(row["created_at"]),
    }


def _to_json(value: Any) -> str:
    """Serialize framework objects to stable JSON for SQLite storage."""

    return json.dumps(to_memory_jsonable(value), sort_keys=True)


def _to_nullable_json(value: Any | None) -> str | None:
    if value is None:
        return None
    return _to_json(value)


def _turn_metrics_from_json(value: Any) -> TurnMetrics:
    if value is None:
        return TurnMetrics()
    payload = from_memory_jsonable(json.loads(str(value)))
    if not isinstance(payload, dict):
        return TurnMetrics()
    allowed_fields = set(TurnMetrics.__dataclass_fields__)
    return TurnMetrics(
        **{key: item for key, item in payload.items() if key in allowed_fields}
    )


def _from_nullable_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return from_memory_jsonable(json.loads(str(value)))
