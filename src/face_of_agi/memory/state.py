"""Persistent state memory M."""

from __future__ import annotations

from typing import Any

from face_of_agi.contracts import (
    AgentTrace,
    ActionSpec,
    ContextDocuments,
    MStateRecord,
    MemoryRecord,
    Observation,
    PostDecisionPredictions,
)
from face_of_agi.memory.sqlite import SQLiteDatabase


class StateMemory:
    """Durable per-game M memory backed by dedicated SQLite state rows."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.database.initialize_schema()

    def write_record(
        self,
        *,
        run_id: str,
        game_id: str,
        kind: str,
        payload: Any,
        step: int | None = None,
    ) -> MemoryRecord:
        """Store one committed runtime record."""

        return self.database.write_record(
            "state_records",
            run_id=run_id,
            game_id=game_id,
            step=step,
            kind=kind,
            payload=payload,
        )

    def list_records(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[MemoryRecord]:
        """List committed runtime records."""

        return self.database.list_records(
            "state_records",
            run_id=run_id,
            game_id=game_id,
        )

    def write_state(
        self,
        *,
        run_id: str,
        game_id: str,
        step: int | None,
        frame_index: int,
        frame_count: int,
        current_observation: Observation,
        chosen_action: ActionSpec,
        contexts: ContextDocuments,
        agent_trace: AgentTrace,
        post_decision_predictions: PostDecisionPredictions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Store one complete frame-after-frame M state."""

        predictions = post_decision_predictions or PostDecisionPredictions()
        return self.database.write_m_state(
            run_id=run_id,
            game_id=game_id,
            step=step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            chosen_action=chosen_action,
            world_context=contexts.world,
            goal_context=contexts.goal,
            agent_context=contexts.agent,
            agent_trace=agent_trace,
            world_prediction=predictions.world_prediction,
            goal_prediction=predictions.goal_prediction,
            metadata=metadata,
        )

    def read_latest_state(self, game_id: str) -> MStateRecord | None:
        """Read the latest complete M state for one game."""

        return self.database.read_latest_m_state(game_id=game_id)

    def list_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List complete M state rows, optionally scoped to one game."""

        return self.database.list_m_states(game_id=game_id)

    def cleanup_keep_latest_per_game(self) -> None:
        """Prune complete M state rows to the newest row for each game."""

        self.database.cleanup_m_states_keep_latest_per_game()

    def clear_states(self) -> None:
        """Delete complete M state rows without touching other memory tables."""

        self.database.clear_m_states()

    def clear_memory_tables(self) -> None:
        """Delete all rows from current and legacy memory tables."""

        self.database.clear_memory_tables()
