"""Persistent state memory M."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionHistoryScoreAdvanceMarker,
    AgentTrace,
    ActionSpec,
    ContextDocuments,
    EnvironmentStepEventRecord,
    FrameControlMode,
    MStateRecord,
    ModelCallEventRecord,
    Observation,
    ObservationRef,
    RoleContext,
    RunMetadataRecord,
    TurnMetrics,
)
from face_of_agi.debug.contracts import ModelInputDebugRecord
from face_of_agi.frames import observation_frame_hash, to_memory_jsonable
from face_of_agi.memory.sqlite import SQLiteDatabase

DEFAULT_FRAME_HASH_CROP_EDGES = (0, 0, 0, 0)


class StateMemory:
    """Durable per-game M memory backed by dedicated SQLite state rows."""

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
        contexts: ContextDocuments,
        agent_trace: AgentTrace,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Store one complete frame-after-frame M state."""

        return self.database.write_m_state(
            run_id=run_id,
            game_id=game_id,
            step=step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            chosen_action=chosen_action,
            agent_context=contexts.agent,
            agent_trace=agent_trace,
            turn_metrics=(
                turn_metrics or TurnMetrics()
            ),
            metadata=metadata,
        )

    def hydrate_contexts_for_game(
        self,
        *,
        game_id: str,
        defaults: ContextDocuments,
    ) -> ContextDocuments:
        """Combine latest game-agnostic K with latest selected-game L contexts."""

        general_contexts = self.read_latest_general_contexts()
        latest_state = self.read_latest_state(game_id)
        return ContextDocuments(
            agent=_hydrated_role_context(
                general_contexts.agent,
                latest_state.agent_context if latest_state is not None else None,
                defaults.agent,
            ),
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
        contexts: ContextDocuments,
        frame_hash_crop_edges: tuple[int, int, int, int] = (
            DEFAULT_FRAME_HASH_CROP_EDGES
        ),
    ) -> MStateRecord:
        """Create the M source row for one frame turn with standard metadata."""

        frame_hash_metadata = _frame_hash_metadata(
            current_observation,
            crop_edges=frame_hash_crop_edges,
        )
        return self.prewrite_state(
            run_id=run_id,
            game_id=game_id,
            step=current_observation.step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            contexts=contexts,
            metadata={
                "turn_id": turn_id,
                "control_mode": asdict(control_mode),
                "prewritten": True,
                **frame_hash_metadata,
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
        contexts: ContextDocuments,
        agent_trace: AgentTrace,
        turn_metrics: TurnMetrics | None = None,
        game_memory: Any | None = None,
        game_memory_updated_this_turn: bool = False,
        action_history_entry: ActionHistoryItem | None = None,
        action_history_score_advance_marker: (
            ActionHistoryScoreAdvanceMarker | None
        ) = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Complete a prewritten frame-turn M row with standard metadata."""

        source = self.read_state_source(state_id)
        if source is None:
            raise RuntimeError(f"unknown M state row: {state_id}")
        source_metadata = dict(source.metadata)
        source_metadata.pop("prewritten", None)
        return self.complete_state(
            state_id=state_id,
            chosen_action=chosen_action,
            contexts=contexts,
            agent_trace=agent_trace,
            turn_metrics=turn_metrics,
            metadata={
                **source_metadata,
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
                "action_history_entry": (
                    _action_history_metadata(action_history_entry)
                    if action_history_entry is not None
                    else None
                ),
                "action_history_score_advance_marker": (
                    _action_history_metadata(action_history_score_advance_marker)
                    if action_history_score_advance_marker is not None
                    else None
                ),
                "game_memory": _game_memory_metadata(
                    game_memory,
                    updated_this_turn=game_memory_updated_this_turn,
                ),
                **(extra_metadata or {}),
            },
        )

    def prewrite_state(
        self,
        *,
        run_id: str,
        game_id: str,
        step: int | None,
        frame_index: int,
        frame_count: int,
        current_observation: Observation,
        contexts: ContextDocuments,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Create the M source row before Agent X acts."""

        return self.database.prewrite_m_state(
            run_id=run_id,
            game_id=game_id,
            step=step,
            frame_index=frame_index,
            frame_count=frame_count,
            current_observation=current_observation,
            agent_context=contexts.agent,
            metadata=metadata,
        )

    def complete_state(
        self,
        *,
        state_id: int,
        chosen_action: ActionSpec,
        contexts: ContextDocuments,
        agent_trace: AgentTrace,
        turn_metrics: TurnMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Complete a prewritten M source row after the frame turn resolves."""

        return self.database.complete_m_state(
            state_id=state_id,
            chosen_action=chosen_action,
            agent_context=contexts.agent,
            agent_trace=agent_trace,
            turn_metrics=(
                turn_metrics or TurnMetrics()
            ),
            metadata=metadata,
        )

    def read_state_source(self, state_id: int) -> MStateRecord | None:
        """Read a source M row by id, including the current incomplete row."""

        return self.database.read_m_state_source(state_id=state_id)

    def read_complete_state_before(
        self,
        *,
        game_id: str,
        state_id: int,
    ) -> MStateRecord | None:
        """Read the newest complete M row before the given state row id."""

        return self.database.read_complete_m_state_before(
            game_id=game_id,
            state_id=state_id,
        )

    def read_recent_agent_game_contexts(
        self,
        *,
        game_id: str,
        run_id: str,
        before_state_id: int | None,
        limit: int,
    ) -> tuple[str, ...]:
        """Return recent same-run complete agent game contexts before a state row."""

        if before_state_id is None or limit <= 0:
            return ()
        return self.database.read_recent_agent_game_contexts_before(
            game_id=game_id,
            run_id=run_id,
            state_id=before_state_id,
            limit=limit,
        )

    def read_agent_game_context_history(
        self,
        *,
        game_id: str,
        run_id: str,
        before_state_id: int | None,
        limit: int,
    ) -> tuple[str, ...]:
        """Return recent same-run complete agent game contexts oldest-to-newest."""

        return tuple(
            reversed(
                self.read_recent_agent_game_contexts(
                    game_id=game_id,
                    run_id=run_id,
                    before_state_id=before_state_id,
                    limit=limit,
                )
            )
        )

    def read_latest_state(self, game_id: str) -> MStateRecord | None:
        """Read the latest complete M state for one game."""

        return self.database.read_latest_m_state(game_id=game_id)

    def read_latest_general_contexts(self) -> ContextDocuments:
        """Read the latest game-agnostic contexts across all games."""

        return self.database.read_latest_general_contexts()

    def update_state_contexts(
        self,
        *,
        state_id: int,
        contexts: ContextDocuments,
    ) -> MStateRecord:
        """Update stored contexts on an existing complete M state row."""

        return self.database.update_m_state_contexts(
            state_id=state_id,
            agent_context=contexts.agent,
        )

    def merge_state_metadata(
        self,
        *,
        state_id: int,
        metadata: dict[str, Any],
    ) -> MStateRecord:
        """Merge metadata into an existing M state row."""

        current = self.read_state_source(state_id)
        if current is None:
            raise RuntimeError(f"unknown M state row: {state_id}")
        return self.database.update_m_state_metadata(
            state_id=state_id,
            metadata={**current.metadata, **metadata},
        )

    def list_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List complete M state rows, optionally scoped to one game."""

        return self.database.list_m_states(game_id=game_id)

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
        """Store one raw provider request for debug inspection."""

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
        """Store normalized provider request captures for one M state row."""

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
        """List raw provider request records for debug inspection."""

        return self.database.list_model_input_debug_records(
            m_state_id=m_state_id,
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
        )

    def write_model_call_event(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int | None,
        role: str,
        provider: str,
        model: str | None,
        event: str,
        status: str,
        queue_wait_seconds: float | None = None,
        duration_seconds: float | None = None,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallEventRecord:
        """Store one model-call lifecycle event."""

        return self.database.write_model_call_event(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            role=role,
            provider=provider,
            model=model,
            event=event,
            status=status,
            queue_wait_seconds=queue_wait_seconds,
            duration_seconds=duration_seconds,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )

    def list_model_call_events(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
        turn_id: int | None = None,
        role: str | None = None,
    ) -> list[ModelCallEventRecord]:
        """List stored model-call lifecycle events."""

        return self.database.list_model_call_events(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            role=role,
        )

    def write_environment_step_event(
        self,
        *,
        run_id: str,
        game_id: str,
        turn_id: int | None,
        step: int | None,
        action: dict[str, Any],
        status: str,
        duration_seconds: float,
        remaining_actions: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentStepEventRecord:
        """Store one environment-step timing event."""

        return self.database.write_environment_step_event(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
            step=step,
            action=action,
            status=status,
            duration_seconds=duration_seconds,
            remaining_actions=remaining_actions,
            metadata=metadata,
        )

    def list_environment_step_events(
        self,
        *,
        run_id: str | None = None,
        game_id: str | None = None,
        turn_id: int | None = None,
    ) -> list[EnvironmentStepEventRecord]:
        """List stored environment-step timing events."""

        return self.database.list_environment_step_events(
            run_id=run_id,
            game_id=game_id,
            turn_id=turn_id,
        )

    def cleanup_keep_latest_per_game(self) -> None:
        """Prune complete M state rows to the newest row for each game."""

        self.database.cleanup_m_states_keep_latest_per_game()

    def clear_states(self) -> None:
        """Delete complete M state rows without touching other memory tables."""

        self.database.clear_m_states()

    def clear_memory_tables(self) -> None:
        """Delete all rows from current memory tables."""

        self.database.clear_memory_tables()


def _hydrated_role_context(
    general_context: RoleContext,
    game_context: RoleContext | None,
    default_context: RoleContext,
) -> RoleContext:
    """Combine cross-game K with selected-game L for one model role."""

    return RoleContext(
        general=general_context.general or default_context.general,
        game=game_context.game if game_context is not None else default_context.game,
    )


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _frame_hash_metadata(
    observation: Observation,
    *,
    crop_edges: tuple[int, int, int, int],
) -> dict[str, Any]:
    try:
        return {
            "current_frame_hash": observation_frame_hash(
                observation,
                crop_edges=crop_edges,
            ),
            "current_frame_hash_crop_edges": crop_edges,
        }
    except Exception as exc:
        return {
            "current_frame_hash_unavailable": {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "current_frame_hash_crop_edges": crop_edges,
        }


def _game_memory_metadata(
    game_memory: Any | None,
    *,
    updated_this_turn: bool,
) -> dict[str, Any]:
    markdown = getattr(game_memory, "markdown", None)
    metadata = getattr(game_memory, "metadata", None)
    available = bool(
        getattr(game_memory, "is_available", lambda: bool(markdown))()
    )
    return {
        "document": markdown if isinstance(markdown, str) else "not available",
        "available": available,
        "updated_this_turn": updated_this_turn,
        "metadata": _dict(metadata),
    }


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
    return to_memory_jsonable(item)
