"""Known-state simulation for repeated ARC frames.

The simulation path is owned by orchestration. It runs after Agent X chooses an
action and before ARC is stepped. Matched rows replay a prior known transition:
the environment step and change-summary model are skipped, while the normal
post-transition memory, historizer, updater, persistence, and action-history
paths still run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from arcengine import GameState

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
    ChangeSummaryElement,
    ContextDocuments,
    FrameTurnContext,
    Observation,
    ObservationRef,
    UpdaterFrameTransitionInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import EnvironmentStepRecorded, FrameTurnCompleted
from face_of_agi.memory import StateMemory
from face_of_agi.models.adapters import (
    AgentContextHistorizerModel,
    OrchestratorAgentModel,
)
from face_of_agi.models.action_coordinates import (
    arc_grid_edges_to_normalized_crop_box,
    normalized_1000_to_arc_grid_coordinate,
)
from face_of_agi.models.memory import GameMemoryModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.orchestration.game_loop.actions import steps
from face_of_agi.orchestration.game_loop.actions.metrics import turn_metrics
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)

SIMULATED_ROW_KEY = "simulated"
SIMULATION_METADATA_KEY = "known_state_simulation"
SIMULATION_CATCHUP_KEY = "known_state_simulation_catchup"
MAX_SIMULATION_STEPS = 64

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]


@dataclass(frozen=True, slots=True)
class KnownStateTransitionEdge:
    """One historical transition that can be replayed without stepping ARC."""

    source_state_id: int
    successor_state_id: int
    source_frame_hash: str
    successor_frame_hash: str
    action: ActionSpec
    successor_observation: Observation
    action_history_entry: ActionHistoryEntry
    score_advance_marker: ActionHistoryScoreAdvanceMarker | None = None


@dataclass(frozen=True, slots=True)
class SimulationCatchupPlan:
    """Real actions used to move live ARC to the simulated endpoint."""

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
    agent: OrchestratorAgentModel,
    agent_context_historizer: AgentContextHistorizerModel | None,
    game_memory_model: GameMemoryModel,
    updater_tasks: UpdaterTaskRegistry,
    tool_runtime_factory: AgentToolRuntimeFactory | None,
    state_memory: StateMemory | None,
    frame_hash_crop_edges: tuple[int, int, int, int],
    debug: DebugBus,
) -> bool:
    """Replay known transitions when Agent X selects a previously seen edge.

    Returns true when the outer state-machine turn was consumed and should
    restart. Returns false when no simulation happened, or when simulation
    caught up successfully and the current Agent X decision should continue
    through the normal real environment-step path.
    """

    if state_memory is None:
        return False
    if not _current_turn_can_simulate(session):
        return False

    current = steps.require_current(session)
    decision = steps.require_decision(session)
    entry_hash = _current_frame_hash(state_memory, current)
    if entry_hash is None:
        return False

    edges = _known_state_transition_edges(
        state_memory,
        game_id=current.game_id,
        run_id=current.run_id,
        before_state_id=current.source_state_id or 0,
    )
    first_edge = _edge_for_action(
        edges,
        frame_hash=entry_hash,
        action=decision.final_action,
        crop_edges=frame_hash_crop_edges,
    )
    if first_edge is None:
        return False

    live_entry_observation = current.observation
    simulated_actions: list[ActionSpec] = []
    last_simulated_state_id: int | None = None
    simulation_started_at = perf_counter()

    while True:
        current = steps.require_current(session)
        decision = steps.require_decision(session)
        current_hash = _current_frame_hash(state_memory, current)
        if current_hash is None:
            return _finish_without_exit_action(
                session,
                state_memory=state_memory,
                debug=debug,
                edges=edges,
                entry_frame_hash=entry_hash,
                expected_frame_hash=entry_hash,
                simulated_actions=tuple(simulated_actions),
                live_entry_observation=live_entry_observation,
                last_simulated_state_id=last_simulated_state_id,
                frame_hash_crop_edges=frame_hash_crop_edges,
                exit_reason="current_frame_hash_unavailable",
                duration_seconds=perf_counter() - simulation_started_at,
            )

        edge = _edge_for_action(
            edges,
            frame_hash=current_hash,
            action=decision.final_action,
            crop_edges=frame_hash_crop_edges,
        )
        if edge is None:
            return _finish_with_exit_action(
                session,
                state_memory=state_memory,
                debug=debug,
                edges=edges,
                entry_frame_hash=entry_hash,
                expected_frame_hash=current_hash,
                simulated_actions=tuple(simulated_actions),
                live_entry_observation=live_entry_observation,
                exit_action=decision.final_action,
                exit_state_id=current.source_state_id,
                frame_hash_crop_edges=frame_hash_crop_edges,
                exit_reason="unknown_action",
                duration_seconds=perf_counter() - simulation_started_at,
            )

        if len(simulated_actions) >= MAX_SIMULATION_STEPS:
            return _finish_with_exit_action(
                session,
                state_memory=state_memory,
                debug=debug,
                edges=edges,
                entry_frame_hash=entry_hash,
                expected_frame_hash=current_hash,
                simulated_actions=tuple(simulated_actions),
                live_entry_observation=live_entry_observation,
                exit_action=decision.final_action,
                exit_state_id=current.source_state_id,
                frame_hash_crop_edges=frame_hash_crop_edges,
                exit_reason="simulation_step_limit_reached",
                duration_seconds=perf_counter() - simulation_started_at,
            )

        turn_started_at = perf_counter()
        _resolve_simulated_next_snapshot(session, edge=edge)
        _run_simulated_post_transition(
            session,
            contexts=contexts,
            agent_context_historizer=agent_context_historizer,
            game_memory_model=game_memory_model,
            updater_tasks=updater_tasks,
            state_memory=state_memory,
            debug=debug,
        )
        current = steps.require_current(session)
        decision = steps.require_decision(session)
        last_simulated_state_id = current.source_state_id
        simulated_actions.append(decision.final_action)
        _emit_simulated_frame_turn_completed(
            session,
            debug=debug,
            turn_started_at=turn_started_at,
        )
        steps.advance(session)

        abort_reason = _catchup_abort_reason(
            edge.successor_observation,
            completed_levels=session.completed_levels,
        )
        if abort_reason is not None:
            return _finish_without_exit_action(
                session,
                state_memory=state_memory,
                debug=debug,
                edges=edges,
                entry_frame_hash=entry_hash,
                expected_frame_hash=edge.successor_frame_hash,
                simulated_actions=tuple(simulated_actions),
                live_entry_observation=live_entry_observation,
                last_simulated_state_id=last_simulated_state_id,
                frame_hash_crop_edges=frame_hash_crop_edges,
                exit_reason=abort_reason,
                duration_seconds=perf_counter() - simulation_started_at,
            )

        if len(simulated_actions) >= MAX_SIMULATION_STEPS:
            return _finish_without_exit_action(
                session,
                state_memory=state_memory,
                debug=debug,
                edges=edges,
                entry_frame_hash=entry_hash,
                expected_frame_hash=edge.successor_frame_hash,
                simulated_actions=tuple(simulated_actions),
                live_entry_observation=live_entry_observation,
                last_simulated_state_id=last_simulated_state_id,
                frame_hash_crop_edges=frame_hash_crop_edges,
                exit_reason="simulation_step_limit_reached",
                duration_seconds=perf_counter() - simulation_started_at,
            )

        steps.enter_frame_turn(
            session,
            contexts=contexts,
            state_memory=state_memory,
            tool_runtime_factory=tool_runtime_factory,
            frame_hash_crop_edges=frame_hash_crop_edges,
            debug=debug,
        )
        steps.decide(
            session,
            agent=agent,
            contexts=contexts,
            debug=debug,
        )


def _current_turn_can_simulate(session: GameLoopSession) -> bool:
    current = session.current
    decision = session.decision
    if current is None or decision is None:
        return False
    if current.source_state_id is None:
        return False
    if current.control_mode is None or not current.control_mode.controllable:
        return False
    if decision.final_action.is_none():
        return False
    return True


def _resolve_simulated_next_snapshot(
    session: GameLoopSession,
    *,
    edge: KnownStateTransitionEdge,
) -> None:
    current = steps.require_current(session)
    decision = steps.require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    next_observation = edge.successor_observation
    next_ref = session.current_ref_for(next_observation)
    session.next_environment_observation = next_observation
    session.next_frame_buffer = (next_observation,)
    session.transition_frame_observations = (current.observation, next_observation)
    session.turn_metrics = turn_metrics(
        actual_next_observation=next_observation,
        trace_cost_seconds=session.trace_cost_seconds,
        cumulative_time_cost=float(session.real_step_count),
    )
    score_advance_marker = steps.build_score_advance_marker(
        previous_score=session.last_observed_cumulative_score,
        new_score=session.turn_metrics.cumulative_score,
    )
    session.next = FrameTurnSnapshot(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id + 1,
        observation=next_observation,
        observation_ref=next_ref,
        source_state_id=None,
        frame_index=0,
        frame_count=1,
        control_mode=None,
        first_observation_ref=current.first_observation_ref,
        previous_observation_ref=current.observation_ref,
        recent_action_history=current.recent_action_history,
    )
    action_history_entry = replace(
        edge.action_history_entry,
        action=decision.final_action,
        action_count=_next_action_count(session),
    )
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=current.observation_ref,
        actual_next_observation_ref=next_ref,
        decision_trace=decision.trace,
        actual_next_observation=next_observation,
        turn_metrics=session.turn_metrics,
        submitted_action=decision.final_action,
        action_history_entry=action_history_entry,
        action_history_score_advance_marker=(
            edge.score_advance_marker or score_advance_marker
        ),
        metadata={
            "controllable": True,
            SIMULATION_METADATA_KEY: {"kind": "known_state_replay"},
        },
    )


def _run_simulated_post_transition(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    agent_context_historizer: AgentContextHistorizerModel | None,
    game_memory_model: GameMemoryModel,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory,
    debug: DebugBus,
) -> None:
    steps.summarize_game_memory(
        session,
        memory_model=game_memory_model,
        debug=debug,
    )
    agent_context_history = steps.summarize_agent_context_history(
        session,
        state_memory=state_memory,
        agent_context_historizer=agent_context_historizer,
        debug=debug,
    )
    steps.run_updaters(
        session,
        contexts=contexts,
        agent_context_history=agent_context_history,
        updater_tasks=updater_tasks,
        state_memory=state_memory,
        debug=debug,
    )
    _persist_simulated_turn(
        session,
        contexts=contexts,
        state_memory=state_memory,
        debug=debug,
    )


def _persist_simulated_turn(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    state_memory: StateMemory,
    debug: DebugBus,
) -> None:
    current = steps.require_current(session)
    update_input = steps.require_update_input(session)
    simulation_metadata = {
        SIMULATED_ROW_KEY: True,
        SIMULATION_METADATA_KEY: {
            "kind": "known_state_replay",
        },
    }
    update_input.metadata = {
        **update_input.metadata,
        **simulation_metadata,
    }
    if current.source_state_id is None:
        raise RuntimeError("simulated turn is missing a source state id")
    state_memory.merge_state_metadata(
        state_id=current.source_state_id,
        metadata=simulation_metadata,
    )
    steps.persist(
        session,
        contexts=contexts,
        state_memory=state_memory,
        debug=debug,
    )


def _emit_simulated_frame_turn_completed(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    turn_started_at: float,
) -> None:
    current = steps.require_current(session)
    decision = steps.require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("completed frame turn is missing control mode")
    debug.emit(
        FrameTurnCompleted(
            run_id=session.config.run_id,
            game_id=session.game_id,
            game_index=session.environment_config.game_index,
            turn_id=current.turn_id,
            env_step=current.observation.step,
            frame_index=current.frame_index,
            frame_count=current.frame_count,
            controllable=current.control_mode.controllable,
            action=decision.final_action,
            turn_duration_seconds=perf_counter() - turn_started_at,
            completed_levels=_completed_levels_after_turn(session),
            remaining_actions=session.remaining_actions,
        )
    )


def _finish_with_exit_action(
    session: GameLoopSession,
    *,
    state_memory: StateMemory,
    debug: DebugBus,
    edges: Sequence[KnownStateTransitionEdge],
    entry_frame_hash: str,
    expected_frame_hash: str,
    simulated_actions: tuple[ActionSpec, ...],
    live_entry_observation: Observation,
    exit_action: ActionSpec,
    exit_state_id: int | None,
    frame_hash_crop_edges: tuple[int, int, int, int],
    exit_reason: str,
    duration_seconds: float,
) -> bool:
    catchup_metadata = _finish_simulation(
        session,
        debug=debug,
        edges=edges,
        entry_frame_hash=entry_frame_hash,
        expected_frame_hash=expected_frame_hash,
        simulated_actions=simulated_actions,
        live_entry_observation=live_entry_observation,
        exit_action=exit_action,
        frame_hash_crop_edges=frame_hash_crop_edges,
        exit_reason=exit_reason,
        duration_seconds=duration_seconds,
    )
    if exit_state_id is not None:
        state_memory.merge_state_metadata(
            state_id=exit_state_id,
            metadata={SIMULATION_CATCHUP_KEY: catchup_metadata},
        )
    if catchup_metadata["successful"]:
        return False
    _clear_current_turn_after_catchup_failure(session)
    return True


def _finish_without_exit_action(
    session: GameLoopSession,
    *,
    state_memory: StateMemory,
    debug: DebugBus,
    edges: Sequence[KnownStateTransitionEdge],
    entry_frame_hash: str,
    expected_frame_hash: str,
    simulated_actions: tuple[ActionSpec, ...],
    live_entry_observation: Observation,
    last_simulated_state_id: int | None,
    frame_hash_crop_edges: tuple[int, int, int, int],
    exit_reason: str,
    duration_seconds: float,
) -> bool:
    catchup_metadata = _finish_simulation(
        session,
        debug=debug,
        edges=edges,
        entry_frame_hash=entry_frame_hash,
        expected_frame_hash=expected_frame_hash,
        simulated_actions=simulated_actions,
        live_entry_observation=live_entry_observation,
        exit_action=None,
        frame_hash_crop_edges=frame_hash_crop_edges,
        exit_reason=exit_reason,
        duration_seconds=duration_seconds,
    )
    if last_simulated_state_id is not None:
        state_memory.merge_state_metadata(
            state_id=last_simulated_state_id,
            metadata={SIMULATION_CATCHUP_KEY: catchup_metadata},
        )
    _clear_current_turn_after_catchup_failure(session)
    return True


def _finish_simulation(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    edges: Sequence[KnownStateTransitionEdge],
    entry_frame_hash: str,
    expected_frame_hash: str,
    simulated_actions: tuple[ActionSpec, ...],
    live_entry_observation: Observation,
    exit_action: ActionSpec | None,
    frame_hash_crop_edges: tuple[int, int, int, int],
    exit_reason: str,
    duration_seconds: float,
) -> dict[str, Any]:
    catchup_plan = _simulation_catchup_plan(
        edges=edges,
        entry_frame_hash=entry_frame_hash,
        simulated_end_frame_hash=expected_frame_hash,
        simulated_actions=simulated_actions,
    )
    catchup_result = _execute_catchup_actions(
        session,
        debug=debug,
        actions=catchup_plan.actions,
        expected_frame_hash=expected_frame_hash,
        live_entry_observation=live_entry_observation,
        frame_hash_crop_edges=frame_hash_crop_edges,
    )
    catchup_action_count = len(catchup_result.submitted_actions)
    simulated_action_count = len(simulated_actions)
    return {
        "successful": catchup_result.successful,
        "expected_frame_hash": expected_frame_hash,
        "actual_frame_hash": catchup_result.actual_frame_hash,
        "frame_hash_crop_edges": frame_hash_crop_edges,
        "simulated_actions": tuple(action.name for action in simulated_actions),
        "catchup_actions": tuple(
            action.name for action in catchup_result.submitted_actions
        ),
        "catchup_source": catchup_plan.source,
        "catchup_source_state_ids": catchup_plan.source_state_ids,
        "catchup_fallback_reason": catchup_plan.fallback_reason,
        "exit_action": exit_action.name if exit_action is not None else None,
        "exit_reason": exit_reason,
        "aborted": catchup_result.aborted,
        "abort_reason": catchup_result.abort_reason,
        "duration_seconds": duration_seconds,
        "simulated_row_count": simulated_action_count,
        "simulated_action_count": simulated_action_count,
        "catchup_action_count": catchup_action_count,
        "saved_environment_action_count": max(
            0,
            simulated_action_count - catchup_action_count,
        ),
    }


def _execute_catchup_actions(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    actions: tuple[ActionSpec, ...],
    expected_frame_hash: str,
    live_entry_observation: Observation,
    frame_hash_crop_edges: tuple[int, int, int, int],
) -> _CatchupExecutionResult:
    latest_observation = live_entry_observation
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
        step_started_at = perf_counter()
        try:
            latest_observation = session.environment.step(action)
        except Exception as exc:
            debug.record_environment_step_event(
                run_id=session.config.run_id,
                game_id=session.game_id,
                turn_id=session.frame_turn_count,
                step=latest_observation.step,
                action=_action_payload(action),
                status="error",
                duration_seconds=perf_counter() - step_started_at,
                remaining_actions=session.remaining_actions,
                metadata={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "boundary": "known_state_simulation_catchup",
                },
            )
            raise
        session.real_step_count += 1
        session.remaining_actions -= 1
        submitted_actions.append(action)
        session.latest_environment_observation = latest_observation
        debug.record_environment_step_event(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=session.frame_turn_count,
            step=latest_observation.step,
            action=_action_payload(action),
            status="success",
            duration_seconds=perf_counter() - step_started_at,
            remaining_actions=session.remaining_actions,
            metadata={
                "next_observation_id": latest_observation.id,
                "boundary": "known_state_simulation_catchup",
            },
        )
        debug.emit(
            EnvironmentStepRecorded(
                action=action,
                next_observation=latest_observation,
                remaining_actions=session.remaining_actions,
            )
        )
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

    if actions:
        frame = latest_observation.frames[-1] if latest_observation.frames else None
        if frame is None:
            frame = latest_observation.frame
        if frame is not None:
            session.frame_buffer = (
                Observation(
                    id=latest_observation.id,
                    step=latest_observation.step,
                    frame=frame,
                    frames=(frame,),
                    raw_frame_data=latest_observation.raw_frame_data,
                    metadata=dict(latest_observation.metadata),
                ),
            )
            session.frame_index = 0
    actual_hash = _observation_hash(
        latest_observation,
        crop_edges=frame_hash_crop_edges,
    )
    return _CatchupExecutionResult(
        successful=actual_hash == expected_frame_hash,
        actual_frame_hash=actual_hash,
        submitted_actions=tuple(submitted_actions),
    )


def _simulation_catchup_plan(
    *,
    edges: Sequence[KnownStateTransitionEdge],
    entry_frame_hash: str,
    simulated_end_frame_hash: str,
    simulated_actions: tuple[ActionSpec, ...],
) -> SimulationCatchupPlan:
    if entry_frame_hash == simulated_end_frame_hash:
        return SimulationCatchupPlan(
            actions=(),
            source="already_at_simulated_endpoint",
        )
    if len(simulated_actions) <= 1:
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

    for _search_step in range(len(simulated_actions)):
        next_branches: list[_CatchupSearchBranch] = []
        expanded_source_state_ids: set[int] = set()
        for branch in active_branches:
            for edge in _edges_from_hash(edges, frame_hash=branch.current_frame_hash):
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
        source_hash = _frame_hash_from_metadata(row.metadata)
        successor_hash = _frame_hash_from_metadata(successor.metadata)
        if source_hash is None or successor_hash is None:
            continue
        history_entry = _transition_history_entry(row.metadata) or (
            _latest_transition_history_entry(successor.metadata)
        )
        if history_entry is None:
            continue
        _require_replay_history_matches_action(
            action=action,
            history_entry=history_entry,
            source_state_id=row.id,
            successor_state_id=successor.id,
        )
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
                action_history_entry=history_entry,
                score_advance_marker=_transition_score_advance_marker(row.metadata),
            )
        )
    return tuple(edges)


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


def _edges_from_hash(
    edges: Sequence[KnownStateTransitionEdge],
    *,
    frame_hash: str,
) -> tuple[KnownStateTransitionEdge, ...]:
    return tuple(
        edge for edge in reversed(edges) if edge.source_frame_hash == frame_hash
    )


def _next_real_row(state_id: int, *, rows: Sequence[Any]) -> Any | None:
    for row in rows:
        if row.id <= state_id:
            continue
        if _is_simulated_row(row.metadata):
            return None
        return row
    return None


def _current_frame_hash(
    state_memory: StateMemory,
    current: FrameTurnSnapshot,
) -> str | None:
    if current.source_state_id is None:
        return None
    source = state_memory.read_state_source(current.source_state_id)
    if source is None:
        raise RuntimeError(f"unknown M state row: {current.source_state_id}")
    return _frame_hash_from_metadata(source.metadata)


def _frame_hash_from_metadata(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("current_frame_hash")
    return value if isinstance(value, str) and value else None


def _observation_hash(
    observation: Observation,
    *,
    crop_edges: tuple[int, int, int, int],
) -> str | None:
    from face_of_agi.frames import observation_frame_hash

    try:
        return observation_frame_hash(observation, crop_edges=crop_edges)
    except Exception:
        return None


def _transition_history_entry(metadata: dict[str, Any]) -> ActionHistoryEntry | None:
    payload = metadata.get("action_history_entry")
    if not isinstance(payload, dict):
        return None
    return _action_history_entry_from_payload(payload)


def _latest_transition_history_entry(
    metadata: dict[str, Any],
) -> ActionHistoryEntry | None:
    raw_history = metadata.get("recent_action_history")
    if not isinstance(raw_history, list):
        return None
    entries = [
        _action_history_entry_from_payload(item)
        for item in raw_history
        if isinstance(item, dict)
        and item.get("type") not in {"score_advance", "game_reset"}
        and isinstance(item.get("action"), dict)
    ]
    return entries[-1] if entries else None


def _transition_score_advance_marker(
    metadata: dict[str, Any],
) -> ActionHistoryScoreAdvanceMarker | None:
    payload = metadata.get("action_history_score_advance_marker")
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "score_advance":
        return None
    return ActionHistoryScoreAdvanceMarker(
        previous_score=_optional_float(payload.get("previous_score")),
        new_score=float(payload.get("new_score")),
        delta=_optional_float(payload.get("delta")),
    )


def _action_history_entry_from_payload(
    payload: dict[str, Any],
) -> ActionHistoryEntry:
    action = _action_from_payload(payload.get("action"))
    if action is None:
        raise RuntimeError("action history entry is missing an action")
    return ActionHistoryEntry(
        action=action,
        controllable=bool(payload.get("controllable")),
        changed_pixel_count=int(payload.get("changed_pixel_count") or 0),
        change_summary=str(payload.get("change_summary") or ""),
        change_elements=tuple(
            _change_summary_element_from_payload(item)
            for item in payload.get("change_elements") or ()
            if isinstance(item, dict)
        ),
        changed_pixel_percent=_optional_float(payload.get("changed_pixel_percent")),
        completed_levels=_optional_int(payload.get("completed_levels")),
        action_count=_optional_int(payload.get("action_count")),
        skipped_intermediate_animation_frame_count=int(
            payload.get("skipped_intermediate_animation_frame_count") or 0
        ),
    )


def _change_summary_element_from_payload(
    payload: dict[str, Any],
) -> ChangeSummaryElement:
    return ChangeSummaryElement(
        element_name=str(payload.get("element_name") or ""),
        element_description=str(payload.get("element_description") or ""),
        element_mutation=str(payload.get("element_mutation") or ""),
    )


def _replayable_action_from_payload(value: Any) -> ActionSpec | None:
    action = _action_from_payload(value)
    if action is None or action.is_none():
        return None
    return action


def _action_from_payload(value: Any) -> ActionSpec | None:
    if isinstance(value, ActionSpec):
        return value
    if not isinstance(value, dict):
        return None
    action_id = value.get("action_id")
    if action_id is None:
        return None
    return ActionSpec(
        action_id=action_id,
        data=_dict_or_none(value.get("data")),
        target=(
            str(value["target"])
            if value.get("target") is not None
            else None
        ),
        target_value=_target_value_from_payload(value.get("target_value")),
    )


def _persisted_actions_match(left: ActionSpec, right: ActionSpec) -> bool:
    if left.name != right.name:
        return False
    if left.name == "ACTION6":
        return (
            _action_data_signature(left.data) == _action_data_signature(right.data)
            and left.target_value == right.target_value
        )
    return (
        _action_data_signature(left.data) == _action_data_signature(right.data)
        and left.target == right.target
        and left.target_value == right.target_value
    )


def _actions_match_for_known_state(
    *,
    historical: ActionSpec,
    current: ActionSpec,
    crop_edges: tuple[int, int, int, int],
) -> bool:
    if historical.name != current.name:
        return False
    if current.name != "ACTION6":
        return _persisted_actions_match(historical, current)
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
    grid_x0 = _normalized_1000_to_arc_grid(x0, "x", crop_edges=crop_edges)
    grid_x1 = _normalized_1000_to_arc_grid(x1, "x", crop_edges=crop_edges)
    grid_y0 = _normalized_1000_to_arc_grid(y0, "y", crop_edges=crop_edges)
    grid_y1 = _normalized_1000_to_arc_grid(y1, "y", crop_edges=crop_edges)
    left, right = sorted((grid_x0, grid_x1))
    top, bottom = sorted((grid_y0, grid_y1))
    return left <= x <= right and top <= y <= bottom


def _normalized_1000_to_arc_grid(
    value: int,
    key: str,
    *,
    crop_edges: tuple[int, int, int, int],
) -> int:
    return normalized_1000_to_arc_grid_coordinate(
        {key: value},
        key,
        crop_box_normalized=arc_grid_edges_to_normalized_crop_box(crop_edges),
    )


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


def _require_replay_history_matches_action(
    *,
    action: ActionSpec,
    history_entry: ActionHistoryEntry,
    source_state_id: int,
    successor_state_id: int,
) -> None:
    if _persisted_actions_match(action, history_entry.action):
        return
    raise RuntimeError(
        "known-state replay edge has inconsistent action history: "
        f"source_state_id={source_state_id} "
        f"successor_state_id={successor_state_id} "
        f"chosen_action={action.name} "
        f"history_action={history_entry.action.name}"
    )


def _observation_from_payload(value: Any) -> Observation:
    if isinstance(value, Observation):
        return value
    if not isinstance(value, dict):
        raise RuntimeError("persisted observation payload must be a mapping")
    frames = value.get("frames") or ()
    return Observation(
        id=str(value.get("id") or ""),
        step=int(value.get("step") or 0),
        frame=value.get("frame"),
        frames=tuple(frames) if isinstance(frames, (list, tuple)) else (),
        raw_frame_data=None,
        metadata=_dict(value.get("metadata")),
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


def _clear_current_turn_after_catchup_failure(session: GameLoopSession) -> None:
    session.current = None
    session.next = None
    session.tool_runtime = None
    session.decision = None
    session.decision_duration_seconds = None
    session.trace_cost_seconds = None
    session.turn_metrics = None
    session.update_input = None
    session.next_environment_observation = None
    session.next_frame_buffer = ()
    session.transition_frame_observations = ()
    session.process_turn = False


def _next_action_count(session: GameLoopSession) -> int:
    action_counts = [
        item.action_count
        for item in session.action_history
        if isinstance(item, ActionHistoryEntry) and item.action_count is not None
    ]
    if action_counts:
        return max(action_counts) + 1
    return session.real_step_count + 1


def _completed_levels_after_turn(session: GameLoopSession) -> int:
    metrics = session.turn_metrics
    if metrics is not None and metrics.cumulative_score is not None:
        return int(metrics.cumulative_score)
    return int(session.completed_levels)


def _is_simulated_row(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(SIMULATED_ROW_KEY))


def _action_data_signature(data: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not data:
        return ()
    return tuple(sorted(data.items()))


def _action_payload(action: ActionSpec) -> dict[str, Any]:
    return {
        "action_id": action.name,
        "data": action.data,
        "target": action.target,
        "target_value": action.target_value,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _target_value_from_payload(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
