"""Compatibility wrapper for learner debug artifacts."""

from __future__ import annotations

from typing import Any

from face_of_agi.memory.sqlite import SQLiteDatabase


class ExperimentalMemory:
    """Small facade over the learner artifact table."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.database.initialize_schema()

    def write_experiment(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        tool_call: Any,
        output_description: Any,
        tool_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store one debug payload as a learner artifact."""

        return self.database.write_learner_artifact(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            kind=f"debug_artifact:{_tool_name(tool_call)}",
            payload={
                "tool_call": tool_call,
                "output_description": output_description,
                "tool_result": tool_result,
            },
            metadata=metadata,
        )

    def read_experiment(self, ref_id: str | int) -> dict[str, Any] | None:
        """Read one artifact by id."""

        for artifact in self.database.list_learner_artifacts():
            if int(artifact["id"]) == int(ref_id):
                return artifact
        return None

    def list_experiments(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List learner artifact rows."""

        return self.database.list_learner_artifacts(
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

        del run_id, max_turns, game_id

    def clear_experiments(self) -> None:
        """Delete all learner artifact rows."""

        self.database.clear_learner_artifacts()


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("tool") or value.get("name") or "unknown")
    return str(getattr(value, "tool", "unknown"))
