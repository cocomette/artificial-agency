"""Persistent online-learner state memory M."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
    FrameControlMode,
    LearnerTurnTrace,
    MStateRecord,
    Observation,
    ObservationRef,
    RunMetadataRecord,
    TurnMetrics,
)
from face_of_agi.debug.contracts import ModelInputDebugRecord
from face_of_agi.memory.sqlite import SQLiteDatabase


class StateMemory:
    """Durable per-game online-learner memory backed by SQLite rows."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.database.initialize_schema()

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
        learner_snapshot: Mapping[str, Any],
        learner_trace: LearnerTurnTrace,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Store one complete online-learner frame turn."""

        return self.database.write_m_state(
            run_id=run_id,
            game_id=game_id,
            step=step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            chosen_action=chosen_action,
            learner_snapshot=dict(learner_snapshot),
            learner_trace=learner_trace,
            turn_metrics=turn_metrics or TurnMetrics(),
            metadata=metadata,
        )

    def prewrite_frame_turn_source(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        current_observation: Observation,
        frame_index: int,
        frame_count: int,
        control_mode: FrameControlMode,
        learner_snapshot: Mapping[str, Any],
    ) -> MStateRecord:
        """Create the source row before the learner acts."""

        return self.database.prewrite_m_state(
            run_id=run_id,
            game_id=game_id,
            step=current_observation.step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            learner_snapshot=dict(learner_snapshot),
            metadata={
                "turn_id": turn_id,
                "control_mode": asdict(control_mode),
                "prewritten": True,
            },
        )

    def complete_frame_turn_state(
        self,
        *,
        state_id: int,
        turn_id: int,
        control_mode: FrameControlMode,
        previous_observation_ref: ObservationRef | None,
        recent_action_history: tuple[ActionHistoryItem, ...],
        chosen_action: ActionSpec,
        learner_snapshot: Mapping[str, Any],
        learner_trace: LearnerTurnTrace,
        turn_metrics: TurnMetrics | None = None,
    ) -> MStateRecord:
        """Complete a prewritten source row after the transition resolves."""

        return self.database.complete_m_state(
            state_id=state_id,
            chosen_action=chosen_action,
            learner_snapshot=dict(learner_snapshot),
            learner_trace=learner_trace,
            turn_metrics=turn_metrics or TurnMetrics(),
            metadata={
                "turn_id": turn_id,
                "control_mode": asdict(control_mode),
                "previous_observation_ref": (
                    asdict(previous_observation_ref)
                    if previous_observation_ref is not None
                    else None
                ),
                "recent_action_history": [
                    _action_history_metadata(item) for item in recent_action_history
                ],
            },
        )

    def read_state_source(self, state_id: int) -> MStateRecord | None:
        """Read a source M row by id, including incomplete rows."""

        return self.database.read_m_state_source(state_id=state_id)

    def read_latest_state(self, game_id: str) -> MStateRecord | None:
        """Read the latest complete M state for one game."""

        return self.database.read_latest_m_state(game_id=game_id)

    def list_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List complete M state rows, optionally scoped to one game."""

        return self.database.list_m_states(game_id=game_id)

    def write_learner_artifact(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int,
        kind: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store one optional learner artifact."""

        return self.database.write_learner_artifact(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            kind=kind,
            payload=payload,
            metadata=metadata,
        )

    def list_learner_artifacts(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored learner artifacts."""

        return self.database.list_learner_artifacts(run_id=run_id, game_id=game_id)

    def write_run_metadata(
        self,
        *,
        run_id: str,
        game_id: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunMetadataRecord:
        """Store one run-level metadata row."""

        return self.database.write_run_metadata(
            run_id=run_id,
            game_id=game_id,
            kind=kind,
            metadata=metadata,
        )

    def list_run_metadata(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
        kind: str | None = None,
    ) -> list[RunMetadataRecord]:
        """List stored run-level metadata rows."""

        return self.database.list_run_metadata(
            run_id=run_id,
            game_id=game_id,
            kind=kind,
        )

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
        """Store one passive debug request record."""

        return self.database.write_model_input_debug_record(
            m_state_id=m_state_id,
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            call_slot=call_slot,
            provider=provider,
            model=model,
            phase=phase,
            attempt=attempt,
            request=request,
            usage=usage,
            metadata=metadata,
        )

    def write_model_input_debug_records(
        self,
        *,
        m_state_id: int,
        run_id: str,
        game_id: str,
        turn_id: int,
        records: Iterable[Mapping[str, Any]],
    ) -> list[ModelInputDebugRecord]:
        """Store normalized passive debug request captures for one M state row."""

        stored: list[ModelInputDebugRecord] = []
        for record in records:
            request = record.get("request")
            stored.append(
                self.write_model_input_debug_record(
                    m_state_id=m_state_id,
                    run_id=run_id,
                    game_id=game_id,
                    turn_id=turn_id,
                    call_slot=str(record.get("call_slot") or "unknown"),
                    provider=str(record.get("provider") or "unknown"),
                    model=(
                        str(record["model"])
                        if record.get("model") is not None
                        else None
                    ),
                    phase=str(record.get("phase") or "unknown"),
                    attempt=_non_negative_int(record.get("attempt")),
                    request=(
                        request if isinstance(request, dict) else {"value": request}
                    ),
                    usage=record.get("usage"),
                    metadata=_dict(record.get("metadata")),
                )
            )
        return stored

    def list_model_input_debug_records(
        self,
        *,
        m_state_id: int | None = None,
        run_id: str | None = None,
        game_id: str | None = None,
        turn_id: int | None = None,
    ) -> list[ModelInputDebugRecord]:
        """List passive debug request records."""

        return self.database.list_model_input_debug_records(
            m_state_id=m_state_id,
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
        )

    def cleanup_keep_latest_per_game(self) -> None:
        """Prune complete M state rows to the newest row for each game."""

        self.database.cleanup_m_states_keep_latest_per_game()

    def clear_states(self) -> None:
        """Delete complete M state rows without touching run metadata."""

        self.database.clear_m_states()

    def clear_memory_tables(self) -> None:
        """Delete all rows from current memory tables."""

        self.database.clear_memory_tables()


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _action_history_metadata(item: ActionHistoryItem) -> dict[str, Any]:
    if isinstance(item, ActionHistoryResetMarker):
        return {
            "type": "game_reset",
            "reason": item.reason,
            "restart_count": item.restart_count,
        }
    if isinstance(item, ActionHistoryScoreAdvanceMarker):
        return {
            "type": "score_advance",
            "previous_score": item.previous_score,
            "new_score": item.new_score,
            "delta": item.delta,
        }
    return asdict(item)
