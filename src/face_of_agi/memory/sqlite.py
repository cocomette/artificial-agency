"""SQLite primitives for the framework memory domains."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from face_of_agi.contracts import (
    ContextDocuments,
    EExperimentRecord,
    LevelSolutionSummaryRecord,
    MStateRecord,
    ObservationRef,
    TurnMetrics,
    RoleContext,
    SamePastStateDetection,
)
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
        "agent_context_json",
        "agent_trace_json",
        "turn_metrics_json",
        "metadata_json",
        "created_at",
    ),
    "e_experiments": (
        "id",
        "game_id",
        "run_id",
        "turn_id",
        "tool_name",
        "source_state_id",
        "tool_call_json",
        "output_description_json",
        "tool_result_json",
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
    "level_solution_summaries": (
        "id",
        "run_id",
        "game_id",
        "completed_level",
        "source_state_ids_json",
        "solution_method",
        "metadata_json",
        "created_at",
    ),
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
        """Create the dedicated memory tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS state_records;
                DROP TABLE IF EXISTS experimental_records;

                CREATE TABLE IF NOT EXISTS m_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    step INTEGER,
                    frame_index INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    current_observation_json TEXT NOT NULL,
                    chosen_action_json TEXT,
                    agent_context_json TEXT NOT NULL,
                    agent_trace_json TEXT,
                    turn_metrics_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS e_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    source_state_id INTEGER,
                    tool_call_json TEXT NOT NULL,
                    output_description_json TEXT NOT NULL,
                    tool_result_json TEXT NOT NULL,
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

                CREATE TABLE IF NOT EXISTS level_solution_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    completed_level INTEGER NOT NULL,
                    source_state_ids_json TEXT NOT NULL,
                    solution_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._require_current_schema(connection)

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
        agent_context: RoleContext,
        agent_trace: Any,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Write one complete M state row for a frame turn."""

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
                        agent_context_json,
                        agent_trace_json,
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
                        _to_json(agent_context),
                        _to_json(agent_trace),
                        _to_json(turn_metrics or TurnMetrics()),
                        _to_json(metadata or {}),
                    ),
                )
                record_id = int(cursor.lastrowid)
                row = connection.execute(
                    "SELECT * FROM m_states WHERE id = ?",
                    (record_id,),
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
        agent_context: RoleContext,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Write the source row for a frame before Agent X acts."""

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
                    agent_context_json,
                    agent_trace_json,
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
                    _to_json(agent_context),
                    _to_json(TurnMetrics()),
                    _to_json(metadata or {}),
                ),
            )
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (record_id,),
            ).fetchone()

        return self._row_to_m_state(row)

    def complete_m_state(
        self,
        *,
        state_id: int,
        chosen_action: Any,
        agent_context: RoleContext,
        agent_trace: Any,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Complete a prewritten M source row after the frame turn resolves."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE m_states
                SET
                    chosen_action_json = ?,
                    agent_context_json = ?,
                    agent_trace_json = ?,
                    turn_metrics_json = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    _to_json(chosen_action),
                    _to_json(agent_context),
                    _to_json(agent_trace),
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
        """Return the newest M state row for a game, if any."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM m_states
                WHERE game_id = ?
                  AND chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_m_state(row)

    def read_latest_general_contexts(self) -> ContextDocuments:
        """Return the newest persisted game-agnostic contexts across all games."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT agent_context_json
                FROM m_states
                WHERE chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return ContextDocuments()
        return ContextDocuments(
            agent=RoleContext(
                general=_role_context_from_json(row["agent_context_json"]).general
            ),
        )

    def update_m_state_contexts(
        self,
        *,
        state_id: int,
        agent_context: RoleContext,
    ) -> MStateRecord:
        """Update stored contexts on an existing complete M state row."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE m_states
                SET
                    agent_context_json = ?
                WHERE id = ?
                """,
                (
                    _to_json(agent_context),
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

    def list_m_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List M state rows, optionally scoped to one game."""

        values: list[str] = []
        clauses = [
            "chosen_action_json IS NOT NULL",
            "agent_trace_json IS NOT NULL",
        ]
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

    def read_m_state_source(self, *, state_id: int) -> MStateRecord | None:
        """Read a source M row by id, including incomplete current rows."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM m_states WHERE id = ?",
                (state_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_m_state(row)

    def read_complete_m_state_before(
        self,
        *,
        game_id: str,
        state_id: int,
    ) -> MStateRecord | None:
        """Return the newest complete M state before a given state id."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM m_states
                WHERE game_id = ?
                  AND id < ?
                  AND chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (game_id, state_id),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_m_state(row)

    def read_recent_agent_context_history_before(
        self,
        *,
        game_id: str,
        run_id: str,
        state_id: int,
        limit: int,
    ) -> tuple[str, ...]:
        """Return recent same-run agent context snapshots before a state id."""

        if limit <= 0:
            return ()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM m_states
                WHERE game_id = ?
                  AND run_id = ?
                  AND id < ?
                  AND chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id DESC
                """,
                (game_id, run_id, state_id),
            ).fetchall()

        snapshots: list[str] = []
        for row in rows:
            snapshot = _metadata_snapshot_text(
                row["metadata_json"],
                key="agent_context_history",
                required_fields=(
                    "probing_strategy",
                    "policy_strategy",
                ),
            )
            if snapshot is not None:
                snapshots.append(snapshot)
            if len(snapshots) == limit:
                break
        return tuple(snapshots)

    def read_agent_strategy_history_between(
        self,
        *,
        game_id: str,
        run_id: str,
        after_state_id: int | None,
        through_state_id: int,
    ) -> tuple[str, ...]:
        """Return same-run strategy snapshots in an M-state id interval."""

        clauses = [
            "game_id = ?",
            "run_id = ?",
            "id <= ?",
            "chosen_action_json IS NOT NULL",
            "agent_trace_json IS NOT NULL",
        ]
        values: list[Any] = [game_id, run_id, through_state_id]
        if after_state_id is not None:
            clauses.append("id > ?")
            values.append(after_state_id)
        where = " AND ".join(clauses)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT metadata_json FROM m_states WHERE {where} ORDER BY id",
                values,
            ).fetchall()

        snapshots: list[str] = []
        for row in rows:
            snapshot = _metadata_snapshot_text(
                row["metadata_json"],
                key="agent_context_history",
                required_fields=(
                    "probing_strategy",
                    "policy_strategy",
                ),
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return tuple(snapshots)

    def write_level_solution_summary(
        self,
        *,
        run_id: str,
        game_id: str,
        completed_level: int,
        source_state_ids: tuple[int, ...],
        solution_method: str,
        metadata: dict[str, Any] | None = None,
    ) -> LevelSolutionSummaryRecord:
        """Write one completed-level solution summary."""

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO level_solution_summaries (
                    run_id,
                    game_id,
                    completed_level,
                    source_state_ids_json,
                    solution_method,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    game_id,
                    completed_level,
                    _to_json(source_state_ids),
                    solution_method,
                    _to_json(metadata or {}),
                ),
            )
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM level_solution_summaries WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._row_to_level_solution_summary(row)

    def read_latest_level_solution_summary(
        self,
        *,
        run_id: str | None = None,
        game_id: str,
    ) -> LevelSolutionSummaryRecord | None:
        """Return the latest level solution summary for one game."""

        clauses = ["game_id = ?"]
        values: list[Any] = [game_id]
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        where = " AND ".join(clauses)
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM level_solution_summaries
                WHERE {where}
                ORDER BY completed_level DESC, id DESC
                LIMIT 1
                """,
                values,
            ).fetchone()
        if row is None:
            return None
        return self._row_to_level_solution_summary(row)

    def read_world_model_context_before(
        self,
        *,
        game_id: str,
        state_id: int,
    ) -> str:
        """Return the newest world-model context before a state id."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM m_states
                WHERE game_id = ?
                  AND id < ?
                  AND chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id DESC
                """,
                (game_id, state_id),
            ).fetchall()

        for row in rows:
            snapshot = _metadata_snapshot_text(
                row["metadata_json"],
                key="world_model_context",
                required_fields=(
                    "world_description",
                    "special_events",
                    "action_effects",
                ),
            )
            if snapshot is not None:
                return snapshot
        return ""

    def read_same_past_state_detections_before(
        self,
        *,
        game_id: str,
        run_id: str,
        state_id: int,
        current_frame_hash: str,
    ) -> tuple[SamePastStateDetection, ...]:
        """Return prior same-run updater plans for exact matching frame hashes."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM m_states
                WHERE game_id = ?
                  AND run_id = ?
                  AND id < ?
                  AND chosen_action_json IS NOT NULL
                  AND agent_trace_json IS NOT NULL
                ORDER BY id
                """,
                (game_id, run_id, state_id),
            ).fetchall()

        detections: list[SamePastStateDetection] = []
        for row in rows:
            detection = _same_past_state_detection_from_row(
                row,
                current_frame_hash=current_frame_hash,
            )
            if detection is not None:
                detections.append(detection)
        return tuple(detections)

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

    def write_e_experiment(
        self,
        *,
        game_id: str,
        run_id: str,
        turn_id: int,
        tool_name: str,
        source_state_id: int,
        tool_call: Any,
        output_description: Any,
        tool_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EExperimentRecord:
        """Write one experimental tool output row."""

        with runtime_timing.span("sqlite.write_e_experiment.execute"):
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO e_experiments (
                        game_id,
                        run_id,
                        turn_id,
                        tool_name,
                        source_state_id,
                        tool_call_json,
                        output_description_json,
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
                        source_state_id,
                        _to_json(tool_call),
                        _to_json(output_description),
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
        """Write one raw provider model-input debug record."""

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
            record_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM model_input_debug_records WHERE id = ?",
                (record_id,),
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
        """List raw provider model-input debug records."""

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
        """Delete all rows from current memory tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                DELETE FROM model_input_debug_records;
                DELETE FROM m_states;
                DELETE FROM e_experiments;
                DELETE FROM level_solution_summaries;
                """
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
            agent_context=_role_context_from_json(row["agent_context_json"]),
            agent_trace=_from_nullable_json(row["agent_trace_json"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
            turn_metrics=_turn_metrics_from_json(
                row["turn_metrics_json"]
            ),
        )

    def _row_to_e_experiment(self, row: sqlite3.Row) -> EExperimentRecord:
        return EExperimentRecord(
            id=int(row["id"]),
            game_id=str(row["game_id"]),
            run_id=str(row["run_id"]),
            turn_id=int(row["turn_id"]),
            tool_name=str(row["tool_name"]),
            source_state_id=int(row["source_state_id"] or 0),
            tool_call=from_memory_jsonable(json.loads(str(row["tool_call_json"]))),
            output_description=from_memory_jsonable(
                json.loads(str(row["output_description_json"]))
            ),
            tool_result=from_memory_jsonable(json.loads(str(row["tool_result_json"]))),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
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

    def _row_to_level_solution_summary(
        self,
        row: sqlite3.Row,
    ) -> LevelSolutionSummaryRecord:
        raw_source_ids = from_memory_jsonable(
            json.loads(str(row["source_state_ids_json"]))
        )
        if not isinstance(raw_source_ids, list):
            raise RuntimeError("stored level summary source ids must be a JSON list")
        return LevelSolutionSummaryRecord(
            id=int(row["id"]),
            run_id=str(row["run_id"]),
            game_id=str(row["game_id"]),
            completed_level=int(row["completed_level"]),
            source_state_ids=tuple(int(item) for item in raw_source_ids),
            solution_method=str(row["solution_method"]),
            metadata=from_memory_jsonable(json.loads(str(row["metadata_json"]))),
            created_at=str(row["created_at"]),
        )

    def _require_current_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
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


def _to_json(value: Any) -> str:
    """Serialize framework objects to stable JSON for SQLite storage."""

    return json.dumps(to_memory_jsonable(value), sort_keys=True)


def _to_nullable_json(value: Any | None) -> str | None:
    """Serialize an optional framework object to stable JSON."""

    if value is None:
        return None
    return _to_json(value)


def _turn_metrics_from_json(value: Any) -> TurnMetrics:
    """Deserialize stored frame-turn metrics."""

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
    """Deserialize an optional JSON object stored in SQLite."""

    if value is None:
        return None
    return from_memory_jsonable(json.loads(str(value)))


def _metadata_snapshot_text(
    raw_metadata: Any,
    *,
    key: str,
    required_fields: tuple[str, ...],
) -> str | None:
    """Return one structured metadata snapshot as stable prompt JSON."""

    metadata = from_memory_jsonable(json.loads(str(raw_metadata)))
    if not isinstance(metadata, dict):
        raise RuntimeError("M state metadata must be a JSON object")
    snapshot = metadata.get(key)
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise RuntimeError(f"M state metadata {key} must be a JSON object")
    missing = [field for field in required_fields if field not in snapshot]
    if missing:
        raise RuntimeError(
            f"M state metadata {key} is missing fields: " + ", ".join(missing)
        )
    return json.dumps(
        {field: snapshot[field] for field in required_fields},
        indent=2,
        ensure_ascii=False,
    )


def _same_past_state_detection_from_row(
    row: sqlite3.Row,
    *,
    current_frame_hash: str,
) -> SamePastStateDetection | None:
    metadata = from_memory_jsonable(json.loads(str(row["metadata_json"])))
    if not isinstance(metadata, dict):
        raise RuntimeError("M state metadata must be a JSON object")
    if metadata.get("current_frame_hash") != current_frame_hash:
        return None

    strategy = metadata.get("agent_context_history")
    if not isinstance(strategy, dict):
        return None
    probing_strategy = strategy.get("probing_strategy")
    policy_strategy = strategy.get("policy_strategy")
    if not isinstance(probing_strategy, str) or not isinstance(policy_strategy, str):
        raise RuntimeError(
            "M state metadata agent_context_history must contain string "
            "probing_strategy and policy_strategy"
        )

    evolution = metadata.get("agent_context_evolution")
    if not isinstance(evolution, dict):
        raise RuntimeError("M state metadata agent_context_evolution must be an object")
    return SamePastStateDetection(
        probing_strategy=probing_strategy,
        policy_strategy=policy_strategy,
        probing_evolution=_optional_str(evolution.get("probing_evolution")),
        policy_evolution=_optional_str(evolution.get("policy_evolution")),
    )

def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RuntimeError("M state metadata evolution fields must be strings")
    return value


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
