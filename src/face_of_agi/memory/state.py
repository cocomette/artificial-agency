"""Persistent state memory M."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionHistoryResetMarker,
    AgentTrace,
    ActionSpec,
    ContextDocuments,
    FrameControlMode,
    LevelSolutionSummaryRecord,
    MStateRecord,
    Observation,
    ObservationRef,
    RoleContext,
    SamePastStateDetection,
    TurnMetrics,
)
from face_of_agi.debug.contracts import ModelInputDebugRecord
from face_of_agi.frames import observation_frame_hash
from face_of_agi.memory.sqlite import SQLiteDatabase


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
            metadata=_with_current_frame_hash(metadata, current_observation),
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
        current_frame_hash: str | None = None,
        current_frame_hash_crop_edges: tuple[int, int, int, int] | None = None,
    ) -> MStateRecord:
        """Create the M source row for one frame turn with standard metadata."""

        frame_hash = current_frame_hash or observation_frame_hash(
            current_observation,
            crop_edges=current_frame_hash_crop_edges,
        )
        hash_metadata: dict[str, Any] = {"current_frame_hash": frame_hash}
        if current_frame_hash_crop_edges is not None:
            hash_metadata["current_frame_hash_crop_edges"] = list(
                current_frame_hash_crop_edges
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
                **hash_metadata,
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
        agent_context_history: dict[str, Any] | None = None,
        agent_context_evolution: dict[str, Any] | None = None,
        world_model_context: dict[str, Any] | None = None,
    ) -> MStateRecord:
        """Complete a prewritten frame-turn M row with standard metadata."""

        source_metadata = _source_metadata(self.read_state_source(state_id))
        metadata = {
            "turn_id": turn_id,
            "control_mode": asdict(control_mode),
            **_current_frame_hash_metadata(source_metadata),
            "previous_observation_ref": (
                asdict(previous_observation_ref)
                if previous_observation_ref is not None
                else None
            ),
            "recent_action_history": [
                _action_history_metadata(item) for item in recent_action_history
            ],
        }
        if agent_context_history is not None:
            metadata["agent_context_history"] = agent_context_history
        if agent_context_evolution is not None:
            metadata["agent_context_evolution"] = agent_context_evolution
        if world_model_context is not None:
            metadata["world_model_context"] = world_model_context
        return self.complete_state(
            state_id=state_id,
            chosen_action=chosen_action,
            contexts=contexts,
            agent_trace=agent_trace,
            turn_metrics=turn_metrics,
            metadata=metadata,
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
            metadata=_with_current_frame_hash(metadata, current_observation),
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

    def read_agent_context_history(
        self,
        *,
        game_id: str,
        run_id: str,
        before_state_id: int | None,
        limit: int,
    ) -> tuple[str, ...]:
        """Return recent same-run agent context snapshots oldest-to-newest."""

        if before_state_id is None or limit <= 0:
            return ()
        return tuple(
            reversed(
                self.database.read_recent_agent_context_history_before(
                    game_id=game_id,
                    run_id=run_id,
                    state_id=before_state_id,
                    limit=limit,
                )
            )
        )

    def read_agent_strategy_history_between(
        self,
        *,
        game_id: str,
        run_id: str,
        after_state_id: int | None,
        through_state_id: int,
    ) -> tuple[str, ...]:
        """Return same-run strategy snapshots in an M-state id interval."""

        return self.database.read_agent_strategy_history_between(
            game_id=game_id,
            run_id=run_id,
            after_state_id=after_state_id,
            through_state_id=through_state_id,
        )

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
        """Store one same-game solution method from a completed level."""

        return self.database.write_level_solution_summary(
            run_id=run_id,
            game_id=game_id,
            completed_level=completed_level,
            source_state_ids=source_state_ids,
            solution_method=solution_method,
            metadata=metadata,
        )

    def read_latest_level_solution_summary(
        self,
        *,
        game_id: str,
        run_id: str | None = None,
    ) -> LevelSolutionSummaryRecord | None:
        """Read the latest same-game level solution method."""

        return self.database.read_latest_level_solution_summary(
            run_id=run_id,
            game_id=game_id,
        )

    def read_world_model_context_before(
        self,
        *,
        game_id: str,
        before_state_id: int | None,
    ) -> str:
        """Return the newest previous world-model context before a state row."""

        if before_state_id is None:
            return ""
        return self.database.read_world_model_context_before(
            game_id=game_id,
            state_id=before_state_id,
        )

    def read_latest_state(self, game_id: str) -> MStateRecord | None:
        """Read the latest complete M state for one game."""

        return self.database.read_latest_m_state(game_id=game_id)

    def read_same_past_state_detections(
        self,
        *,
        game_id: str,
        run_id: str,
        before_state_id: int | None,
        current_frame_hash: str,
    ) -> tuple[SamePastStateDetection, ...]:
        """Return prior same-run rows with the same frame hash and updater plan."""

        if before_state_id is None:
            return ()
        return self.database.read_same_past_state_detections_before(
            game_id=game_id,
            run_id=run_id,
            state_id=before_state_id,
            current_frame_hash=current_frame_hash,
        )

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

    def list_states(self, *, game_id: str | None = None) -> list[MStateRecord]:
        """List complete M state rows, optionally scoped to one game."""

        return self.database.list_m_states(game_id=game_id)

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


def _with_current_frame_hash(
    metadata: dict[str, Any] | None,
    observation: Observation,
    *,
    crop_edges: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    if metadata is not None and isinstance(metadata.get("current_frame_hash"), str):
        return dict(metadata)
    result = {
        **(metadata or {}),
        "current_frame_hash": observation_frame_hash(observation, crop_edges=crop_edges),
    }
    if crop_edges is not None:
        result["current_frame_hash_crop_edges"] = list(crop_edges)
    return result


def _source_metadata(record: MStateRecord | None) -> dict[str, Any]:
    if record is None:
        raise RuntimeError("cannot complete M state row without its source metadata")
    return record.metadata


def _current_frame_hash_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    frame_hash = metadata.get("current_frame_hash")
    if not isinstance(frame_hash, str) or not frame_hash:
        raise RuntimeError("prewritten M state metadata is missing current_frame_hash")
    hash_metadata: dict[str, Any] = {"current_frame_hash": frame_hash}
    crop_edges = metadata.get("current_frame_hash_crop_edges")
    if crop_edges is not None:
        hash_metadata["current_frame_hash_crop_edges"] = crop_edges
    return hash_metadata


def _action_history_metadata(item: ActionHistoryItem) -> dict[str, Any]:
    if isinstance(item, ActionHistoryResetMarker):
        return {
            "type": "game_reset",
            "reason": item.reason,
            "restart_count": item.restart_count,
        }
    return asdict(item)
