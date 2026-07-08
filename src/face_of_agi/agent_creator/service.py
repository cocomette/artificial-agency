"""Runtime service for non-blocking shared agent-role creation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from face_of_agi.agent_creator.contracts import (
    AgentCreatorBatchItem,
    AgentCreatorGameRequest,
    AgentRoleSnapshot,
    AgentStrategySnapshot,
    ClaimedAgentCreatorBatch,
)
from face_of_agi.agent_creator.defaults import (
    default_agent_roles,
    default_general_agent_system_prompt,
)
from face_of_agi.agent_creator.store import AgentCreatorStore
from face_of_agi.agent_creator.mutations import RoleMutationToolExecutor
from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionSpec,
    Observation,
)
from face_of_agi.memory import SQLiteDatabase, StateMemory
from face_of_agi.models.agent_creator.contracts import (
    AgentCreatorInput,
    CreatorOrchestratorModel,
    RoleAuthorModel,
)

LOGGER = logging.getLogger(__name__)


class AgentCreatorService:
    """Shared role-revision service with one queued-game batch worker."""

    def __init__(
        self,
        *,
        store: AgentCreatorStore,
        creator_model: CreatorOrchestratorModel,
        role_author_model: RoleAuthorModel | None = None,
        batch_size: int,
        max_tool_calls: int = 4,
        max_roles: int = 8,
        strategy_history_window: int = 10,
    ) -> None:
        if batch_size < 1:
            raise ValueError("agent creator batch_size must be at least 1")
        if max_tool_calls < 0:
            raise ValueError("agent creator max_tool_calls must be non-negative")
        if max_roles < 1:
            raise ValueError("agent creator max_roles must be at least 1")
        if strategy_history_window < 0:
            raise ValueError(
                "agent creator strategy_history_window must be non-negative"
            )
        self.store = store
        self.creator_model = creator_model
        self.role_author_model = role_author_model or creator_model
        self.batch_size = batch_size
        self.max_tool_calls = max_tool_calls
        self.max_roles = max_roles
        self.strategy_history_window = strategy_history_window
        self.store.initialize_schema()
        self.store.seed_defaults(
            roles=default_agent_roles(),
            general_system_prompt=default_general_agent_system_prompt(),
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Future[None] | None = None
        self._lock = Lock()

    def latest_snapshot(self) -> AgentRoleSnapshot:
        """Return the latest completed active role projection."""

        snapshot = self.store.read_latest_complete_role_snapshot()
        if snapshot is None:
            raise RuntimeError("agent creator store has no active role projection")
        return snapshot

    def enqueue_game(
        self,
        *,
        run_id: str,
        game_id: str,
        memory_database_path: str | Path,
    ) -> None:
        """Queue one game and trigger a background update if ready."""

        self.store.enqueue_game_request(
            run_id=run_id,
            game_id=game_id,
            memory_database_path=str(memory_database_path),
        )
        self._submit_if_ready()

    def close(self) -> None:
        """Wait for the active background update and stop the executor."""

        with self._lock:
            future = self._active_future
        if future is not None:
            future.result()
        self._executor.shutdown(wait=True)

    def _submit_if_ready(self) -> None:
        with self._lock:
            if self._active_future is not None:
                return
            batch = self.store.claim_full_batch(batch_size=self.batch_size)
            if batch is None:
                return
            self._active_future = self._executor.submit(
                self._process_batch_and_resubmit,
                batch,
            )

    def _process_batch_and_resubmit(self, batch: ClaimedAgentCreatorBatch) -> None:
        try:
            self._process_batch(batch)
        finally:
            with self._lock:
                self._active_future = None
            self._submit_if_ready()

    def _process_batch(self, batch: ClaimedAgentCreatorBatch) -> None:
        snapshot = self.latest_snapshot()
        run = self.store.create_creator_run(
            request_ids=batch.request_ids,
            max_tool_calls=self.max_tool_calls,
        )
        try:
            batch_items = tuple(
                _batch_item_from_request(
                    request,
                    snapshot,
                    strategy_history_window=self.strategy_history_window,
                )
                for request in batch.requests
            )
            tool_executor = RoleMutationToolExecutor(
                store=self.store,
                role_author=self.role_author_model,
                run_id=run.id,
                roles=snapshot.roles,
                general_system_prompt=snapshot.general_system_prompt,
                max_tool_calls=self.max_tool_calls,
                max_roles=self.max_roles,
            )
            response = self.creator_model.run_creator(
                AgentCreatorInput(
                    batch_items=batch_items,
                    current_roles=snapshot.roles,
                    general_system_prompt=snapshot.general_system_prompt,
                    metadata={
                        "previous_role_snapshot_id": snapshot.id,
                        "request_ids": batch.request_ids,
                        "max_roles": self.max_roles,
                    },
                ),
                max_tool_calls=self.max_tool_calls,
            )
            tool_executor.execute_plan(response.mutations)
            self.store.complete_creator_run(run.id)
            self.store.mark_batch_complete(batch.request_ids)
        except Exception as exc:
            LOGGER.error("agent creator batch failed", exc_info=True)
            self.store.fail_creator_run(run.id, error=str(exc))
            self.store.mark_batch_failed(batch.request_ids, error=str(exc))


def _batch_item_from_request(
    request: AgentCreatorGameRequest,
    snapshot: AgentRoleSnapshot,
    *,
    strategy_history_window: int,
) -> AgentCreatorBatchItem:
    memory = StateMemory(SQLiteDatabase(request.memory_database_path))
    state = memory.read_latest_complete_state_for_run(
        game_id=request.game_id,
        run_id=request.run_id,
    )
    if state is None:
        raise RuntimeError(
            "agent creator could not find a completed M row for "
            f"run={request.run_id!r} game={request.game_id!r}"
        )
    return AgentCreatorBatchItem(
        run_id=request.run_id,
        game_id=request.game_id,
        role_snapshot_id=snapshot.id,
        state_id=state.id,
        strategy_history=_strategy_history(
            memory=memory,
            game_id=request.game_id,
            run_id=request.run_id,
            through_state_id=state.id,
            window=strategy_history_window,
        ),
        current_observation=_observation_from_state(state.current_observation),
        action_history=_action_history_from_metadata(
            state.metadata.get("agent_creator_action_history"),
        ),
        roles=snapshot.roles,
        general_system_prompt=snapshot.general_system_prompt,
        world_model_context=_dict(state.metadata.get("world_model_context")),
        metadata={
            "source": "latest_complete_m_state",
            "request_id": request.id,
            "memory_database_path": request.memory_database_path,
        },
    )


def _strategy_history(
    *,
    memory: StateMemory,
    game_id: str,
    run_id: str,
    through_state_id: int,
    window: int,
) -> tuple[AgentStrategySnapshot, ...]:
    if window < 0:
        raise ValueError("agent creator strategy_history_window must be non-negative")
    if window == 0:
        return ()
    history = memory.read_agent_strategy_history_between(
        game_id=game_id,
        run_id=run_id,
        after_state_id=None,
        through_state_id=through_state_id,
    )
    return tuple(_strategy_snapshot(item) for item in history[-window:])


def _strategy_snapshot(value: Any) -> AgentStrategySnapshot:
    if isinstance(value, AgentStrategySnapshot):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict) and "strategy" in loaded:
            return AgentStrategySnapshot(
                role=str(loaded.get("role") or "agent"),
                strategy=str(loaded.get("strategy") or ""),
            )
        return AgentStrategySnapshot(role="agent", strategy=value)
    if isinstance(value, dict):
        return AgentStrategySnapshot(
            role=str(value.get("role") or "agent"),
            strategy=str(value.get("strategy") or value),
        )
    return AgentStrategySnapshot(role="agent", strategy=str(value))


def _observation_from_state(value: Any) -> Observation:
    if isinstance(value, Observation):
        return value
    if not isinstance(value, dict):
        raise RuntimeError("M state current_observation must be an object")
    frames = value.get("frames") or ()
    return Observation(
        id=str(value.get("id", "")),
        step=int(value.get("step", 0) or 0),
        frame=value.get("frame"),
        frames=tuple(frames) if isinstance(frames, (list, tuple)) else (),
        raw_frame_data=value.get("raw_frame_data"),
        metadata=_dict(value.get("metadata")),
    )


def _action_history_from_metadata(value: Any) -> tuple[ActionHistoryItem, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeError("agent_creator_action_history metadata must be a list")
    return tuple(_action_history_item_from_metadata(item) for item in value)


def _action_history_item_from_metadata(value: Any) -> ActionHistoryItem:
    if isinstance(value, ActionHistoryEntry | ActionHistoryResetMarker):
        return value
    if not isinstance(value, dict):
        raise RuntimeError("agent creator action history item must be an object")
    if value.get("type") == "game_reset":
        return ActionHistoryResetMarker(
            reason=str(value.get("reason", "")),
            restart_count=int(value.get("restart_count", 0) or 0),
        )
    action_payload = value.get("action")
    action = (
        action_payload
        if isinstance(action_payload, ActionSpec)
        else _action_from_payload(action_payload)
    )
    return ActionHistoryEntry(
        action=action,
        controllable=bool(value.get("controllable", False)),
        changed_pixel_count=float(value.get("changed_pixel_count", 0) or 0),
        change_summary=str(value.get("change_summary", "")),
        completed_levels=_optional_int(value.get("completed_levels")),
        action_count=_optional_int(value.get("action_count")),
        action_mode=(
            str(value["action_mode"])
            if value.get("action_mode") is not None
            else None
        ),
        skipped_intermediate_animation_frame_count=int(
            value.get("skipped_intermediate_animation_frame_count", 0) or 0
        ),
        animation_frame_count=_optional_int(value.get("animation_frame_count")),
        avg_changed_pixel_count=(
            float(value["avg_changed_pixel_count"])
            if value.get("avg_changed_pixel_count") is not None
            else None
        ),
    )


def _action_from_payload(value: Any) -> ActionSpec:
    if isinstance(value, dict):
        return ActionSpec(
            action_id=str(value.get("action_id", "")),
            data=value.get("data") if isinstance(value.get("data"), dict) else None,
            target=(
                str(value["target"])
                if value.get("target") is not None
                else None
            ),
        )
    return ActionSpec(action_id=str(value or ""))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
