"""Temporary experimental memory E."""

from __future__ import annotations

from typing import Any

from face_of_agi.contracts import (
    EExperimentRecord,
    Observation,
    ToolCall,
    ToolResult,
)
from face_of_agi.memory.sqlite import SQLiteDatabase


class ExperimentalMemory:
    """Rolling experimental memory E backed by dedicated SQLite rows."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.database.initialize_schema()

    def write_experiment(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        tool_call: ToolCall,
        output_description: Observation,
        tool_result: ToolResult,
        metadata: dict[str, Any] | None = None,
    ) -> EExperimentRecord:
        """Store one tool-produced output frame in rolling E memory."""

        return self.database.write_e_experiment(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            tool_name=tool_call.tool,
            source_state_id=tool_call.source_state_id,
            tool_call=tool_call,
            output_description=output_description,
            tool_result=tool_result,
            metadata=metadata,
        )

    def read_experiment(self, ref_id: str | int) -> EExperimentRecord | None:
        """Read one E row by the id used in experimental observation refs."""

        return self.database.read_e_experiment(ref_id=ref_id)

    def list_experiments(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[EExperimentRecord]:
        """List dedicated E experiment rows."""

        return self.database.list_e_experiments(
            run_id=run_id,
            game_id=game_id,
        )

    def cleanup_keep_latest_turns_per_game(
        self,
        *,
        run_id: str,
        max_turns: int,
        game_id: str | None = None,
    ) -> None:
        """Prune E to a rolling turn buffer for each game in one run."""

        self.database.cleanup_e_experiments_keep_latest_turns_per_game(
            run_id=run_id,
            game_id=game_id,
            max_turns=max_turns,
        )

    def clear_experiments(self) -> None:
        """Delete all dedicated E experiment rows."""

        self.database.clear_e_experiments()
