"""Known-state simulation loop for repeated ARC frames."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from time import perf_counter
from typing import Any, Sequence

from arcengine import GameState

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryResetMarker,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    FrameControlMode,
    FrameTurnContext,
    MStateRecord,
    Observation,
    ObservationRef,
    TurnMetrics,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    EnvironmentStepRecorded,
    KnownStateSimulationCompleted,
)
from face_of_agi.frames import observation_frame_hash
from face_of_agi.memory import StateMemory
from face_of_agi.models.arc_grid_crop import normalized_1000_to_arc_grid
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.models.compacter import AgentCompacterModel, AgentCompacterSummary
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_agent_context_update,
    agent_context_strategy_snapshot,
    compact_agent_context,
    previous_level_summary_text,
)
from face_of_agi.orchestration.game_loop.actions.steps import (
    LEVEL_SOLVED_RESET_NOTICE,
    _latest_completed_level,
    _updater_history_input,
    _store_solved_level_compacter_summary,
)
from face_of_agi.orchestration.game_loop.helpers import (
    bundle_frame_observations,
    updater_action_decision,
)
from face_of_agi.orchestration.game_loop.session import (
    GameLoopSession,
)

SIMULATION_METADATA_KEY = "known_state_simulation"
SIMULATED_ROW_KEY = "simulated"
SIMULATION_CATCHUP_KEY = "known_state_simulation_catchup"
MAX_SIMULATION_STEPS = 64


@dataclass(frozen=True, slots=True)
class KnownStateTransitionEdge:
    """One real historical transition usable as a known-state simulation edge."""

    source_state_id: int
    successor_state_id: int
    source_frame_hash: str
    successor_frame_hash: str
    action: ActionSpec
    successor_observation: Observation
    action_history_entries: tuple[ActionHistoryEntry, ...]


@dataclass(frozen=True, slots=True)
class SimulationCatchupPlan:
    """Concrete real-environment actions used to reach the simulated endpoint."""

    actions: tuple[ActionSpec, ...]
    source: str
    source_state_ids: tuple[int, ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CatchupSearchBranch:
    current_frame_hash: str
    actions: tuple[ActionSpec, ...]
    source_state_ids: tuple[int, ...]
    visited_source_state_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class _CatchupExecutionResult:
    successful: bool
    actual_frame_hash: str | None
    submitted_actions: tuple[ActionSpec, ...]
    aborted: bool = False
    abort_reason: str | None = None


def maybe_run_known_state_simulation(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    compacter: AgentCompacterModel,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> bool:
    """Run known-state simulation after updater P selected a first action.

    Returns true when the current real frame turn was consumed by simulation and
    the caller should restart the outer loop without submitting an environment
    action for the prewritten current row.
    """

    if state_memory is None:
        return False
    if session.environment_config.updater_actions_window != 1:
        return False
    current = session.current
    if current is None or current.source_state_id is None:
        return False
    if current.control_mode is None or not current.control_mode.controllable:
        return False
    if session.pending_game_over_reset or session.pending_terminal_stop:
        return False
    if len(session.queued_updater_actions) != 1:
        return False

    source = state_memory.read_state_source(current.source_state_id)
    if source is None:
        return False
    entry_hash = _frame_hash_from_metadata(source.metadata)
    crop_edges = _crop_edges_from_metadata(source.metadata)
    edges = _known_state_transition_edges(
        state_memory,
        game_id=current.game_id,
        run_id=current.run_id,
        before_state_id=current.source_state_id,
    )
    if not _edge_for_action(
        edges,
        frame_hash=entry_hash,
        action=session.queued_updater_actions[0],
        crop_edges=crop_edges,
    ):
        return False

    _append_pending_update_action_history(session)
    simulated_actions: list[ActionSpec] = []
    current_observation = current.observation
    current_observation_ref = current.observation_ref
    previous_observation_ref = current.previous_observation_ref
    current_source_state_id = current.source_state_id
    current_hash = entry_hash
    next_action = session.queued_updater_actions[0]
    turn_id = current.turn_id
    exit_reason = "unknown_action"
    steps_run = 0
    simulation_started_at = perf_counter()

    while True:
        if steps_run >= MAX_SIMULATION_STEPS:
            exit_reason = "simulation_step_limit_reached"
            frame_context = _simulation_frame_context(
                session=session,
                turn_id=turn_id,
                current_observation=current_observation,
                current_observation_ref=current_observation_ref,
                current_source_state_id=current_source_state_id,
                previous_observation_ref=previous_observation_ref,
                control_mode=current.control_mode,
            )
            trace = updater_action_decision(
                frame_context=frame_context,
                queued_action=next_action,
            ).trace
            exit_state = _complete_simulated_source_row(
                session,
                state_memory=state_memory,
                contexts=contexts,
                frame_context=frame_context,
                turn_id=turn_id,
                action=next_action,
                trace=trace,
                edge=None,
                exit_reason=exit_reason,
            )
            catchup_metadata = _finish_simulation(
                session,
                debug=debug,
                turn_id=turn_id,
                simulated_actions=tuple(simulated_actions),
                catchup_plan=_simulation_catchup_plan(
                    edges=edges,
                    entry_frame_hash=entry_hash,
                    simulated_end_frame_hash=current_hash,
                    simulated_actions=tuple(simulated_actions),
                ),
                exit_action=next_action,
                expected_frame_hash=current_hash,
                crop_edges=crop_edges,
                exit_reason=exit_reason,
                duration_seconds=perf_counter() - simulation_started_at,
            )
            _persist_simulation_catchup_metadata(
                state_memory=state_memory,
                state_id=exit_state.id,
                catchup_metadata=catchup_metadata,
            )
            return True

        frame_context = _simulation_frame_context(
            session=session,
            turn_id=turn_id,
            current_observation=current_observation,
            current_observation_ref=current_observation_ref,
            current_source_state_id=current_source_state_id,
            previous_observation_ref=previous_observation_ref,
            control_mode=current.control_mode,
        )
        trace = updater_action_decision(
            frame_context=frame_context,
            queued_action=next_action,
        ).trace
        edge = _edge_for_action(
            edges,
            frame_hash=current_hash,
            action=next_action,
            crop_edges=crop_edges,
        )
        replay_history_entries = (
            () if edge is None else _simulation_action_history_entries(session, edge)
        )
        exit_state = _complete_simulated_source_row(
            session,
            state_memory=state_memory,
            contexts=contexts,
            frame_context=frame_context,
            turn_id=turn_id,
            action=next_action,
            trace=trace,
            edge=edge,
            exit_reason=(None if edge is not None else exit_reason),
            action_history_entries=replay_history_entries,
        )

        if edge is None:
            catchup_metadata = _finish_simulation(
                session,
                debug=debug,
                turn_id=turn_id,
                simulated_actions=tuple(simulated_actions),
                catchup_plan=_simulation_catchup_plan(
                    edges=edges,
                    entry_frame_hash=entry_hash,
                    simulated_end_frame_hash=current_hash,
                    simulated_actions=tuple(simulated_actions),
                ),
                exit_action=next_action,
                expected_frame_hash=current_hash,
                crop_edges=crop_edges,
                exit_reason=exit_reason,
                duration_seconds=perf_counter() - simulation_started_at,
            )
            _persist_simulation_catchup_metadata(
                state_memory=state_memory,
                state_id=exit_state.id,
                catchup_metadata=catchup_metadata,
            )
            return True

        simulated_actions.append(next_action)
        session.action_history.extend(replay_history_entries)
        steps_run += 1

        previous_observation_ref = current_observation_ref
        current_observation = edge.successor_observation
        current_observation_ref = ObservationRef(
            memory="state",
            id=current_observation.id,
        )
        current_hash = edge.successor_frame_hash
        turn_id += 1
        current_source = state_memory.prewrite_frame_turn_source(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=turn_id,
            current_observation=current_observation,
            frame_index=0,
            frame_count=1,
            control_mode=current.control_mode,
            contexts=contexts,
            current_frame_hash=current_hash,
            current_frame_hash_crop_edges=crop_edges,
        )
        current_source_state_id = current_source.id
        next_action = _next_simulated_action(
            session,
            contexts=contexts,
            updater_tasks=updater_tasks,
            compacter=compacter,
            state_memory=state_memory,
            debug=debug,
            frame_context=_simulation_frame_context(
                session=session,
                turn_id=turn_id,
                current_observation=current_observation,
                current_observation_ref=current_observation_ref,
                current_source_state_id=current_source_state_id,
                previous_observation_ref=previous_observation_ref,
                control_mode=current.control_mode,
            ),
            current_action_item_count=len(replay_history_entries),
            turn_id=turn_id,
        )

def _simulation_catchup_plan(
    *,
    edges: Sequence[KnownStateTransitionEdge],
    entry_frame_hash: str,
    simulated_end_frame_hash: str,
    simulated_actions: tuple[ActionSpec, ...],
) -> SimulationCatchupPlan:
    """Return the shortest historical action path to the simulated endpoint."""

    nb_sim = len(simulated_actions)
    if entry_frame_hash == simulated_end_frame_hash:
        return SimulationCatchupPlan(
            actions=(),
            source="already_at_simulated_endpoint",
        )
    if nb_sim <= 1:
        return SimulationCatchupPlan(
            actions=simulated_actions,
            source="direct_simulated_path",
        )

    active_branches = (
        _CatchupSearchBranch(
            current_frame_hash=entry_frame_hash,
            actions=(),
            source_state_ids=(),
            visited_source_state_ids=frozenset(),
        ),
    )
    previously_expanded_source_state_ids: set[int] = set()

    for _search_step in range(nb_sim):
        next_branches: list[_CatchupSearchBranch] = []
        expanded_source_state_ids: set[int] = set()

        for branch in active_branches:
            for edge in _edges_from_hash(
                edges,
                frame_hash=branch.current_frame_hash,
            ):
                if edge.source_state_id in branch.visited_source_state_ids:
                    continue
                if edge.source_state_id in previously_expanded_source_state_ids:
                    continue

                actions = (*branch.actions, edge.action)
                source_state_ids = (*branch.source_state_ids, edge.source_state_id)
                if edge.successor_frame_hash == simulated_end_frame_hash:
                    return SimulationCatchupPlan(
                        actions=actions,
                        source="historical_graph",
                        source_state_ids=source_state_ids,
                    )

                next_branches.append(
                    _CatchupSearchBranch(
                        current_frame_hash=edge.successor_frame_hash,
                        actions=actions,
                        source_state_ids=source_state_ids,
                        visited_source_state_ids=(
                            branch.visited_source_state_ids | {edge.source_state_id}
                        ),
                    )
                )
                expanded_source_state_ids.add(edge.source_state_id)

        if not next_branches:
            return SimulationCatchupPlan(
                actions=simulated_actions,
                source="simulated_path_fallback",
                fallback_reason="no_historical_path",
            )

        previously_expanded_source_state_ids.update(expanded_source_state_ids)
        active_branches = tuple(next_branches)

    return SimulationCatchupPlan(
        actions=simulated_actions,
        source="simulated_path_fallback",
        fallback_reason="search_limit_reached",
    )


def _next_simulated_action(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    compacter: AgentCompacterModel,
    state_memory: StateMemory,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    current_action_item_count: int,
    turn_id: int,
) -> ActionSpec:
    environment_config = session.environment_config
    action_history = tuple(
        session.action_history[session.compacter_action_history_start_index :]
    )
    strategy_history = tuple(session.strategy_history_buffer)
    compact_windowed_action_history = action_history[
        -environment_config.compacter_action_history_window:
    ] if environment_config.compacter_action_history_window else ()
    summary = compact_agent_context(
        state_memory=state_memory,
        frame_context=frame_context,
        compacter=compacter,
        current_observation=frame_context.current_observation,
        action_history=compact_windowed_action_history,
        strategy_history=strategy_history,
        allowed_actions=frame_context.control_mode.allowed_actions,
        turn_id=turn_id,
        debug=debug,
    )
    debug.capture_model_inputs(frame_context, turn_id, compacter)
    _store_simulation_compacter_context(session, summary)
    history_input = _updater_history_input(
        action_history=action_history,
        strategy_history=strategy_history,
        current_action_item_count=current_action_item_count,
        max_buffer_items=environment_config.updater_context_history_window,
        previous_completed_level=(
            _latest_completed_level(
                tuple(
                    session.action_history[
                        : session.compacter_action_history_start_index
                    ]
                )
            )
            or 0
        ),
    )
    previous_level_summary = previous_level_summary_text(
        state_memory=state_memory,
        run_id=session.config.run_id,
        game_id=session.game_id,
    )
    if history_input.level_completed:
        _store_solved_level_compacter_summary(
            session,
            state_memory=state_memory,
            frame_context=frame_context,
            action_history=action_history,
            summary=summary,
            next_action_history_start_index=len(session.action_history),
        )
        previous_level_summary = summary.previous_strategy_summary
    session.strategy_history_buffer = list(history_input.agent_strategy_history)
    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=updater_tasks,
        debug=debug,
        frame_context=frame_context,
        current_observation=frame_context.current_observation,
        action_history=history_input.agent_action_history,
        allowed_action_source=frame_context.control_mode.allowed_actions,
        compacter_context=session.compacter_context_summary,
        previous_game_context_history=history_input.agent_strategy_history,
        previous_level_summary=previous_level_summary,
        reset_notice=(
            LEVEL_SOLVED_RESET_NOTICE if history_input.level_completed else ""
        ),
        turn_id=turn_id,
        updater_actions_window=environment_config.updater_actions_window,
    )
    snapshot = agent_context_strategy_snapshot(contexts)
    session.agent_context_strategy_snapshot = snapshot
    session.strategy_history_buffer.append(
        json.dumps(snapshot, indent=2, ensure_ascii=False)
    )
    if len(result.next_actions) != 1:
        raise RuntimeError("known-state simulation requires one updater action")
    return result.next_actions[0]


def _append_pending_update_action_history(session: GameLoopSession) -> None:
    update_input = session.update_input
    if update_input is None:
        return
    if update_input.action_history_entries:
        session.action_history.extend(update_input.action_history_entries)
        return
    if update_input.action_history_entry is not None:
        session.action_history.append(update_input.action_history_entry)


def _simulation_action_history_entries(
    session: GameLoopSession,
    edge: KnownStateTransitionEdge,
) -> tuple[ActionHistoryEntry, ...]:
    next_action_count = _next_level_action_count(session)
    entries: list[ActionHistoryEntry] = []
    for entry in edge.action_history_entries:
        if entry.controllable:
            entries.append(replace(entry, action_count=next_action_count))
            session.level_action_count = next_action_count
            next_action_count += 1
        else:
            entries.append(replace(entry, action_count=None))
    return tuple(entries)


def _next_level_action_count(session: GameLoopSession) -> int:
    if session.level_action_count > 0:
        return session.level_action_count + 1
    action_counts = _current_level_action_counts(session)
    if action_counts:
        return max(action_counts) + 1
    return 1


def _current_level_action_counts(session: GameLoopSession) -> tuple[int, ...]:
    action_counts: list[int] = []
    previous_completed_levels = 0
    for item in session.action_history:
        if isinstance(item, ActionHistoryResetMarker):
            action_counts = []
            previous_completed_levels = session.completed_levels
            continue
        if not isinstance(item, ActionHistoryEntry):
            continue
        completed_levels = item.completed_levels
        completes_level = (
            completed_levels is not None
            and completed_levels > previous_completed_levels
        )
        if item.action_count is not None and not completes_level:
            action_counts.append(item.action_count)
        if completed_levels is not None and completed_levels > previous_completed_levels:
            previous_completed_levels = completed_levels
            action_counts = []
    return tuple(action_counts)


def _store_simulation_compacter_context(
    session: GameLoopSession,
    summary: AgentCompacterSummary,
) -> None:
    session.compacter_context_summary = summary
    session.compacter_context = {
        "world_description": summary.world_description,
        "special_events": summary.special_events,
        "action_effects": dict(summary.action_effects),
        "previous_actions_summary": summary.previous_actions_summary,
        "previous_strategy_summary": summary.previous_strategy_summary,
    }


def _complete_simulated_source_row(
    session: GameLoopSession,
    *,
    state_memory: StateMemory,
    contexts: ContextDocuments,
    frame_context: FrameTurnContext,
    turn_id: int,
    action: ActionSpec,
    trace: AgentTrace,
    edge: KnownStateTransitionEdge | None,
    exit_reason: str | None,
    action_history_entries: tuple[ActionHistoryEntry, ...] = (),
) -> MStateRecord:
    metadata: dict[str, Any] = {
        SIMULATED_ROW_KEY: True,
        SIMULATION_METADATA_KEY: {
            "kind": "known_state_replay",
            "matched_transition_state_id": (
                edge.source_state_id if edge is not None else None
            ),
            "successor_state_id": edge.successor_state_id if edge is not None else None,
            "exit_reason": exit_reason,
        },
    }
    state = state_memory.complete_frame_turn_state(
        state_id=frame_context.current_source_state_id or 0,
        turn_id=turn_id,
        control_mode=frame_context.control_mode,
        previous_observation_ref=frame_context.previous_observation_ref,
        recent_action_history=(
            *frame_context.recent_action_history,
            *action_history_entries,
        ),
        chosen_action=action,
        contexts=contexts,
        agent_trace=trace,
        turn_metrics=TurnMetrics(
            time_cost=float(session.real_step_count),
            trace_cost=0.0,
            cumulative_score=float(session.completed_levels),
        ),
        agent_context_history=agent_context_strategy_snapshot(contexts),
        compacter_context=session.compacter_context,
        extra_metadata=metadata,
    )
    session.state_record_ids.append(state.id)
    return state


def _finish_simulation(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    turn_id: int,
    simulated_actions: tuple[ActionSpec, ...],
    catchup_plan: SimulationCatchupPlan,
    exit_action: ActionSpec | None,
    expected_frame_hash: str,
    crop_edges: tuple[int, int, int, int],
    exit_reason: str,
    duration_seconds: float,
) -> dict[str, Any]:
    catchup_actions = catchup_plan.actions
    simulated_action_count = len(simulated_actions)
    simulated_row_count = simulated_action_count + 1
    catchup_result = _execute_catchup_actions(
        session,
        debug=debug,
        actions=catchup_actions,
        expected_frame_hash=expected_frame_hash,
        crop_edges=crop_edges,
    )
    submitted_catchup_actions = catchup_result.submitted_actions
    catchup_action_count = len(submitted_catchup_actions)
    saved_environment_action_count = max(
        0,
        simulated_action_count - catchup_action_count,
    )
    session.queued_updater_actions = (
        ()
        if catchup_result.aborted
        else (exit_action,)
        if exit_action is not None
        else ()
    )
    catchup_metadata = {
        "successful": catchup_result.successful,
        "expected_frame_hash": expected_frame_hash,
        "actual_frame_hash": catchup_result.actual_frame_hash,
        "simulated_actions": tuple(action.name for action in simulated_actions),
        "catchup_actions": tuple(
            action.name for action in submitted_catchup_actions
        ),
        "catchup_source": catchup_plan.source,
        "catchup_source_state_ids": catchup_plan.source_state_ids,
        "catchup_fallback_reason": catchup_plan.fallback_reason,
        "exit_action": exit_action.name if exit_action is not None else None,
        "exit_reason": exit_reason,
        "aborted": catchup_result.aborted,
        "abort_reason": catchup_result.abort_reason,
        "simulated_row_count": simulated_row_count,
        "simulated_action_count": simulated_action_count,
        "catchup_action_count": catchup_action_count,
        "saved_environment_action_count": saved_environment_action_count,
    }
    session.turn_metadata[SIMULATION_CATCHUP_KEY] = catchup_metadata
    debug.emit(
        KnownStateSimulationCompleted(
            run_id=session.config.run_id,
            game_id=session.game_id,
            game_index=session.environment_config.game_index,
            turn_id=turn_id,
            duration_seconds=duration_seconds,
            simulated_row_count=simulated_row_count,
            simulated_action_count=simulated_action_count,
            catchup_action_count=catchup_action_count,
            saved_environment_action_count=saved_environment_action_count,
        )
    )
    session.frame_turn_count = turn_id
    session.current = None
    session.next = None
    session.previous_observation = None
    session.previous_observation_ref = None
    session.last_decision = None
    session.tool_runtime = None
    session.decision = None
    session.decision_duration_seconds = None
    session.trace_cost_seconds = None
    session.turn_metrics = None
    session.update_input = None
    session.next_environment_observation = None
    session.next_frame_buffer = ()
    session.compacter_context = None
    session.compacter_context_summary = None
    session.agent_context_strategy_snapshot = None
    session.process_turn = False
    return catchup_metadata


def _persist_simulation_catchup_metadata(
    *,
    state_memory: StateMemory,
    state_id: int,
    catchup_metadata: dict[str, Any],
) -> None:
    state_memory.merge_state_metadata(
        state_id=state_id,
        metadata={SIMULATION_CATCHUP_KEY: catchup_metadata},
    )


def _execute_catchup_actions(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    actions: tuple[ActionSpec, ...],
    expected_frame_hash: str,
    crop_edges: tuple[int, int, int, int],
) -> _CatchupExecutionResult:
    latest_observation = _current_live_observation(session)
    latest_frame = latest_observation
    submitted_actions: list[ActionSpec] = []
    abort_reason = _catchup_abort_reason(
        latest_observation,
        completed_levels=session.completed_levels,
    )
    if abort_reason is not None:
        return _CatchupExecutionResult(
            successful=False,
            actual_frame_hash=None,
            submitted_actions=(),
            aborted=True,
            abort_reason=abort_reason,
        )

    for action in actions:
        session.real_step_count += 1
        latest_observation = session.environment.step(action)
        submitted_actions.append(action)
        session.remaining_actions -= 1
        debug.emit(
            EnvironmentStepRecorded(
                action=action,
                next_observation=latest_observation,
                remaining_actions=session.remaining_actions,
            )
        )
        session.latest_environment_observation = latest_observation
        abort_reason = _catchup_abort_reason(
            latest_observation,
            completed_levels=session.completed_levels,
        )
        if abort_reason is not None:
            session.frame_buffer = ()
            session.frame_index = 0
            return _CatchupExecutionResult(
                successful=False,
                actual_frame_hash=None,
                submitted_actions=tuple(submitted_actions),
                aborted=True,
                abort_reason=abort_reason,
            )
        latest_frame = bundle_frame_observations(latest_observation)[-1]

    if actions:
        session.latest_environment_observation = latest_observation
        session.frame_buffer = (latest_frame,)
        session.frame_index = 0
    actual_hash = observation_frame_hash(latest_frame, crop_edges=crop_edges)
    return _CatchupExecutionResult(
        successful=bool(actual_hash == expected_frame_hash),
        actual_frame_hash=actual_hash,
        submitted_actions=tuple(submitted_actions),
    )


def _catchup_abort_reason(
    observation: Observation,
    *,
    completed_levels: int,
) -> str | None:
    raw_frame_data = observation.raw_frame_data or observation.metadata.get(
        "raw_frame_data"
    )
    state = getattr(raw_frame_data, "state", None)
    if _is_game_state(state, GameState.GAME_OVER):
        return "game_over"
    if _is_game_state(state, GameState.WIN):
        return "game_win"
    levels_completed = getattr(raw_frame_data, "levels_completed", None)
    if levels_completed is None:
        levels_completed = observation.metadata.get("levels_completed")
    if levels_completed is not None and int(levels_completed) > completed_levels:
        return "level_completed"
    if observation.frame is None and not observation.frames:
        return "missing_frame"
    return None


def _is_game_state(value: Any, expected: GameState) -> bool:
    if value == expected:
        return True
    name = getattr(value, "name", None)
    if name == expected.name:
        return True
    raw_value = getattr(value, "value", value)
    return raw_value == expected.value


def _current_live_observation(session: GameLoopSession) -> Observation:
    current = session.current
    if current is not None:
        return current.observation
    if session.frame_buffer:
        return session.frame_buffer[session.frame_index]
    return session.latest_environment_observation


def _simulation_frame_context(
    *,
    session: GameLoopSession,
    turn_id: int,
    current_observation: Observation,
    current_observation_ref: ObservationRef,
    current_source_state_id: int,
    previous_observation_ref: ObservationRef | None,
    control_mode: FrameControlMode,
) -> FrameTurnContext:
    first_ref = session.first_observation_ref or current_observation_ref
    return FrameTurnContext(
        run_id=session.config.run_id,
        game_id=session.game_id,
        first_observation_ref=first_ref,
        current_observation_ref=current_observation_ref,
        current_observation=current_observation,
        current_source_state_id=current_source_state_id,
        frame_index=0,
        frame_count=1,
        control_mode=control_mode,
        previous_observation_ref=previous_observation_ref,
        recent_action_history=tuple(session.action_history),
    )


def _known_state_transition_edges(
    state_memory: StateMemory,
    *,
    game_id: str,
    run_id: str,
    before_state_id: int,
) -> tuple[KnownStateTransitionEdge, ...]:
    rows = [
        row
        for row in state_memory.list_states(game_id=game_id)
        if row.run_id == run_id and row.id < before_state_id
    ]
    real_rows = [row for row in rows if not _is_simulated_row(row.metadata)]
    edges: list[KnownStateTransitionEdge] = []
    for row in real_rows:
        successor = _next_real_row(row.id, rows=rows)
        if successor is None:
            continue
        action = _replayable_action_from_payload(row.chosen_action)
        if action is None:
            continue
        if _replayable_action_from_payload(successor.chosen_action) is None:
            continue
        history_entries = _latest_transition_history_entries(successor.metadata)
        _require_replay_history_matches_action(
            action=action,
            history_entries=history_entries,
            source_state_id=row.id,
            successor_state_id=successor.id,
        )
        source_hash = _frame_hash_from_metadata(row.metadata)
        successor_hash = _frame_hash_from_metadata(successor.metadata)
        edges.append(
            KnownStateTransitionEdge(
                source_state_id=row.id,
                successor_state_id=successor.id,
                source_frame_hash=source_hash,
                successor_frame_hash=successor_hash,
                action=action,
                successor_observation=_observation_from_payload(
                    successor.current_observation
                ),
                action_history_entries=history_entries,
            )
        )
    return tuple(edges)


def _replayable_action_from_payload(value: Any) -> ActionSpec | None:
    action = _action_from_payload(value)
    if action is None or action.is_none():
        return None
    return action


def _require_replay_history_matches_action(
    *,
    action: ActionSpec,
    history_entries: tuple[ActionHistoryEntry, ...],
    source_state_id: int,
    successor_state_id: int,
) -> None:
    if not history_entries:
        return
    replay_action = history_entries[0].action
    if _persisted_actions_match(action, replay_action):
        return
    raise RuntimeError(
        "known-state replay edge has inconsistent action history: "
        f"source_state_id={source_state_id} "
        f"successor_state_id={successor_state_id} "
        f"chosen_action={action.name} "
        f"history_action={replay_action.name}"
    )


def _persisted_actions_match(left: ActionSpec, right: ActionSpec) -> bool:
    if left.name != right.name:
        return False
    if left.name != "ACTION6":
        return _action_signature(left) == _action_signature(right)
    return (
        _action_data_signature(left) == _action_data_signature(right)
        and left.target_value == right.target_value
    )


def _next_real_row(state_id: int, *, rows: Sequence[Any]) -> Any | None:
    for row in rows:
        if row.id <= state_id:
            continue
        if _is_simulated_row(row.metadata):
            return None
        return row
    return None


def _edge_for_action(
    edges: Sequence[KnownStateTransitionEdge],
    *,
    frame_hash: str,
    action: ActionSpec,
    crop_edges: tuple[int, int, int, int],
) -> KnownStateTransitionEdge | None:
    for edge in reversed(edges):
        if edge.source_frame_hash != frame_hash:
            continue
        if _actions_match_for_known_state(
            historical=edge.action,
            current=action,
            crop_edges=crop_edges,
        ):
            return edge
    return None


def _actions_match_for_known_state(
    *,
    historical: ActionSpec,
    current: ActionSpec,
    crop_edges: tuple[int, int, int, int],
) -> bool:
    if historical.name != current.name:
        return False
    if current.name != "ACTION6":
        return _action_signature(historical) == _action_signature(current)
    return _action6_matches_for_known_state(
        historical=historical,
        current=current,
        crop_edges=crop_edges,
    )


def _action6_matches_for_known_state(
    *,
    historical: ActionSpec,
    current: ActionSpec,
    crop_edges: tuple[int, int, int, int],
) -> bool:
    if historical.target_value is None or current.target_value is None:
        return False
    if historical.target_value != current.target_value:
        return False
    if historical.data is None or current.target_bbox is None:
        return False
    x = _arc_grid_coordinate(historical.data, "x")
    y = _arc_grid_coordinate(historical.data, "y")
    return _action6_bbox_contains_grid_coordinate(
        current.target_bbox,
        crop_edges=crop_edges,
        x=x,
        y=y,
    )


def _action6_bbox_contains_grid_coordinate(
    bbox: tuple[int, int, int, int],
    *,
    crop_edges: tuple[int, int, int, int],
    x: int,
    y: int,
) -> bool:
    x0, y0, x1, y1 = bbox
    grid_x0 = normalized_1000_to_arc_grid(
        float(x0),
        "x",
        crop_edges=crop_edges,
    )
    grid_x1 = normalized_1000_to_arc_grid(
        float(x1),
        "x",
        crop_edges=crop_edges,
    )
    grid_y0 = normalized_1000_to_arc_grid(
        float(y0),
        "y",
        crop_edges=crop_edges,
    )
    grid_y1 = normalized_1000_to_arc_grid(
        float(y1),
        "y",
        crop_edges=crop_edges,
    )
    left, right = sorted((grid_x0, grid_x1))
    top, bottom = sorted((grid_y0, grid_y1))
    return left <= x <= right and top <= y <= bottom


def _arc_grid_coordinate(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"ACTION6 historical data missing numeric {key}")
    numeric = float(value)
    if not numeric.is_integer():
        raise RuntimeError(f"ACTION6 historical data {key} must be an integer")
    coordinate = int(numeric)
    if not 0 <= coordinate < 64:
        raise RuntimeError(f"ACTION6 historical data {key} must be in 0..63")
    return coordinate


def _edges_from_hash(
    edges: Sequence[KnownStateTransitionEdge],
    *,
    frame_hash: str,
) -> tuple[KnownStateTransitionEdge, ...]:
    return tuple(
        edge for edge in reversed(edges) if edge.source_frame_hash == frame_hash
    )


def _latest_transition_history_entries(
    metadata: dict[str, Any],
) -> tuple[ActionHistoryEntry, ...]:
    raw_history = metadata.get("recent_action_history")
    if not isinstance(raw_history, list):
        return ()
    entries = [
        entry
        for item in raw_history
        if (entry := _action_history_entry_from_payload(item)) is not None
    ]
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if not entry.controllable:
            continue
        selected = [entry]
        for follow in entries[index + 1 :]:
            if follow.controllable:
                break
            selected.append(follow)
        return tuple(selected)
    return ()


def _action_history_entry_from_payload(value: Any) -> ActionHistoryEntry | None:
    if not isinstance(value, dict) or value.get("type") == "game_reset":
        return None
    action = _action_from_payload(value.get("action"))
    if action is None:
        return None
    return ActionHistoryEntry(
        action=action,
        controllable=bool(value.get("controllable")),
        changed_pixel_count=float(value.get("changed_pixel_count") or 0.0),
        change_summary=str(value.get("change_summary") or ""),
        change_elements=tuple(
            _change_element_from_payload(item)
            for item in value.get("change_elements") or ()
            if isinstance(item, dict)
        ),
        completed_levels=(
            int(value["completed_levels"])
            if value.get("completed_levels") is not None
            else None
        ),
        action_count=(
            int(value["action_count"]) if value.get("action_count") is not None else None
        ),
        skipped_intermediate_animation_frame_count=int(
            value.get("skipped_intermediate_animation_frame_count") or 0
        ),
        animation_frame_count=(
            int(value["animation_frame_count"])
            if value.get("animation_frame_count") is not None
            else None
        ),
        avg_changed_pixel_count=(
            float(value["avg_changed_pixel_count"])
            if value.get("avg_changed_pixel_count") is not None
            else None
        ),
    )


def _change_element_from_payload(value: dict[str, Any]) -> ChangeSummaryElement:
    return ChangeSummaryElement(
        element_name=str(value.get("element_name") or ""),
        element_description=str(value.get("element_description") or ""),
        element_mutation=str(value.get("element_mutation") or ""),
    )


def _observation_from_payload(value: Any) -> Observation:
    if not isinstance(value, dict):
        raise RuntimeError("stored observation payload must be a mapping")
    return Observation(
        id=str(value.get("id") or ""),
        step=int(value.get("step") or 0),
        frame=value.get("frame"),
        frames=tuple(value.get("frames") or ()),
        raw_frame_data=value.get("raw_frame_data"),
        metadata=dict(value.get("metadata") or {}),
    )


def _action_from_payload(value: Any) -> ActionSpec | None:
    if not isinstance(value, dict):
        return None
    if value.get("action_id") is None:
        return None
    data = value.get("data")
    return ActionSpec(
        action_id=str(value["action_id"]),
        data=(data if isinstance(data, dict) else None),
        target=(str(value["target"]) if value.get("target") is not None else None),
        target_value=(
            int(value["target_value"]) if value.get("target_value") is not None else None
        ),
    )


def _action_signature(action: ActionSpec) -> tuple[str, str, str]:
    return (
        action.name,
        _action_data_signature(action),
        action.target or "",
    )


def _action_data_signature(action: ActionSpec) -> str:
    return json.dumps(action.data or {}, sort_keys=True)


def _frame_hash_from_source(
    state_memory: StateMemory,
    source_state_id: int | None,
) -> str:
    if source_state_id is None:
        return ""
    source = state_memory.read_state_source(source_state_id)
    if source is None:
        return ""
    return _frame_hash_from_metadata(source.metadata)


def _compacter_context_text(session: GameLoopSession) -> str:
    if session.compacter_context is None:
        return "not available"
    return json.dumps(session.compacter_context, indent=2, ensure_ascii=False)


def _frame_hash_from_metadata(metadata: dict[str, Any]) -> str:
    frame_hash = metadata.get("current_frame_hash")
    if not isinstance(frame_hash, str) or not frame_hash:
        raise RuntimeError("M state metadata is missing current_frame_hash")
    return frame_hash


def _crop_edges_from_metadata(metadata: dict[str, Any]) -> tuple[int, int, int, int]:
    raw_edges = metadata.get("current_frame_hash_crop_edges")
    if not isinstance(raw_edges, (list, tuple)) or len(raw_edges) != 4:
        return (0, 0, 0, 0)
    return tuple(int(edge) for edge in raw_edges)


def _is_simulated_row(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(SIMULATED_ROW_KEY))
