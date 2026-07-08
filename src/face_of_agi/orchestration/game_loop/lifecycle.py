"""Run lifecycle actions and decisions for the game-loop state machine."""

from __future__ import annotations

import time
from time import perf_counter

from arcengine import GameState

from face_of_agi.contracts import (
    ActionHistoryResetMarker,
    ContextDocuments,
    GameRunResult,
    RuntimeConfig,
)
from face_of_agi.environment.adapter import EnvironmentAdapter
from face_of_agi.environment.config import EnvironmentConfig
from face_of_agi.memory import StateMemory
from face_of_agi.models.level_summary import (
    LevelSolutionSummarizerModel,
    LevelSolutionSummaryInput,
)
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import ModelCallCompleted, RunStarted, RunStopped
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_general_context_updates,
)
from face_of_agi.orchestration.game_loop.session import GameLoopSession

RUNTIME_DEADLINE_REACHED = "runtime_deadline_reached"
LEVEL_LIMIT_REACHED = "level_limit_reached"


def start_run(
    *,
    config: RuntimeConfig,
    environment: EnvironmentAdapter,
    environment_config: EnvironmentConfig,
    contexts: ContextDocuments,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> GameLoopSession:
    """Initialize one game run before entering the frame-turn loop."""

    if environment_config.game_id is None:
        raise RuntimeError("environment config is missing the resolved game_id")

    selected_game_id = environment.select_game_by_id(environment_config.game_id)
    if environment_config.use_learned_contexts and state_memory is not None:
        hydrated = state_memory.hydrate_contexts_for_game(
            game_id=selected_game_id,
            defaults=contexts,
        )
        contexts.agent = hydrated.agent
    observation = environment.reset()
    debug.emit(
        RunStarted(
            run_id=config.run_id,
            game_id=selected_game_id,
            config=environment_config,
        )
    )
    return GameLoopSession(
        config=config,
        environment=environment,
        environment_config=environment_config,
        game_id=selected_game_id,
        latest_environment_observation=observation,
        remaining_actions=environment_config.max_actions_per_level,
    )


def check_lifecycle(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None = None,
    level_solution_summarizer: LevelSolutionSummarizerModel | None = None,
    debug: DebugBus | None = None,
) -> None:
    """Handle lifecycle states before processing a frame turn."""

    info = session.environment.get_info()
    session.current_info = info
    state = info.state

    if state == GameState.WIN:
        completed_levels = max(session.completed_levels, info.levels_completed)
        if completed_levels > session.completed_levels:
            _summarize_completed_levels(
                session,
                state_memory=state_memory,
                level_solution_summarizer=level_solution_summarizer,
                debug=debug,
                previous_completed_levels=session.completed_levels,
                completed_levels=completed_levels,
            )
            session.completed_levels = completed_levels
            session.last_completed_levels = completed_levels
        stop_session(
            session,
            stop_reason="game_end",
            completed_levels=info.levels_completed,
            last_state=state,
        )
        return

    completed_levels = max(session.completed_levels, info.levels_completed)
    if completed_levels > session.completed_levels:
        _summarize_completed_levels(
            session,
            state_memory=state_memory,
            level_solution_summarizer=level_solution_summarizer,
            debug=debug,
            previous_completed_levels=session.completed_levels,
            completed_levels=completed_levels,
        )
        session.completed_levels = completed_levels
        session.last_completed_levels = completed_levels
        session.remaining_actions = session.environment_config.max_actions_per_level
        session.queued_updater_actions = ()
        session.queued_updater_mode = None

    if _level_limit_reached(session, completed_levels):
        stop_session(
            session,
            stop_reason=LEVEL_LIMIT_REACHED,
            completed_levels=completed_levels,
            last_state=state,
        )
        return

    if session.remaining_actions <= 0:
        stop_session(
            session,
            stop_reason="action_limit_reached",
            completed_levels=session.completed_levels,
            last_state=state,
        )
        return

    if state == GameState.GAME_OVER:
        if session.previous_observation is not None and session.last_decision is not None:
            session.pending_game_over_reset = True
            session.queued_updater_actions = ()
            session.queued_updater_mode = None
            session.real_actions = tuple(info.available_actions) or tuple(
                session.environment.get_action_space()
            )
            return
        reset_after_game_over(session)
        return

    session.real_actions = tuple(info.available_actions) or tuple(
        session.environment.get_action_space()
    )


def _summarize_completed_levels(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None,
    level_solution_summarizer: LevelSolutionSummarizerModel | None,
    debug: DebugBus | None,
    previous_completed_levels: int,
    completed_levels: int,
) -> None:
    if state_memory is None or level_solution_summarizer is None:
        return
    if not session.state_record_ids:
        return
    latest_state_id = session.state_record_ids[-1]
    latest_summary = state_memory.read_latest_level_solution_summary(
        run_id=session.config.run_id,
        game_id=session.game_id,
    )
    after_state_id = None
    if latest_summary is not None and latest_summary.source_state_ids:
        after_state_id = latest_summary.source_state_ids[-1]
    source_state_ids = tuple(
        state_id
        for state_id in session.state_record_ids
        if after_state_id is None or state_id > after_state_id
    )
    if not source_state_ids:
        return
    strategy_history = state_memory.read_agent_strategy_history_between(
        game_id=session.game_id,
        run_id=session.config.run_id,
        after_state_id=after_state_id,
        through_state_id=latest_state_id,
    )
    for completed_level in range(previous_completed_levels + 1, completed_levels + 1):
        started_at = perf_counter()
        summary = level_solution_summarizer.summarize_level_solution(
            LevelSolutionSummaryInput(
                run_id=session.config.run_id,
                game_id=session.game_id,
                completed_level=completed_level,
                strategy_history=strategy_history,
                metadata={
                    "source_state_ids": source_state_ids,
                    "previous_completed_levels": previous_completed_levels,
                },
            )
        )
        if debug is not None:
            debug.emit(
                ModelCallCompleted(
                    role="level_summary",
                    duration_seconds=perf_counter() - started_at,
                )
            )
        state_memory.write_level_solution_summary(
            run_id=session.config.run_id,
            game_id=session.game_id,
            completed_level=completed_level,
            source_state_ids=source_state_ids,
            solution_method=summary.solution_method,
            metadata=summary.metadata,
        )


def _level_limit_reached(session: GameLoopSession, completed_levels: int) -> bool:
    cap = session.environment_config.max_levels_per_game
    return cap is not None and completed_levels >= cap


def check_runtime_deadline(session: GameLoopSession) -> bool:
    """Stop the run if the runtime-level deadline has been reached."""

    deadline = session.config.deadline_monotonic
    if deadline is None or time.monotonic() < deadline:
        return False
    stop_for_runtime_deadline(session)
    return True


def stop_for_runtime_deadline(session: GameLoopSession) -> None:
    """Record a clean runtime deadline stop without more environment actions."""

    completed_levels = session.completed_levels
    last_state = None
    if session.current_info is not None:
        completed_levels = max(completed_levels, session.current_info.levels_completed)
        last_state = session.current_info.state
    stop_session(
        session,
        stop_reason=RUNTIME_DEADLINE_REACHED,
        completed_levels=completed_levels,
        last_state=last_state,
    )


def stop_session(
    session: GameLoopSession,
    *,
    stop_reason: str,
    completed_levels: int,
    last_state: GameState | None,
) -> None:
    """Set the single terminal result consumed by the run exit path."""

    session.terminal_result = GameRunResult(
        run_id=session.config.run_id,
        game_id=session.game_id,
        initial_observation_ref=session.first_observation_ref,
        decision=session.last_decision,
        state_record_ids=tuple(session.state_record_ids),
        stop_reason=stop_reason,
        step_count=session.real_step_count,
        completed_levels=completed_levels,
        last_state=last_state,
    )
    session.running = False
    session.process_turn = False


def reset_after_game_over(session: GameLoopSession) -> None:
    """Reset ARC after GAME_OVER while keeping the run loop alive."""

    session.latest_environment_observation = session.environment.reset()
    session.game_start_turn_id = session.frame_turn_count + 1
    session.game_start_reason = "game_over_reset"
    session.game_restart_count += 1
    session.action_history.append(
        ActionHistoryResetMarker(
            reason=session.game_start_reason,
            restart_count=session.game_restart_count,
        )
    )
    session.previous_observation_ref = None
    session.previous_observation = None
    session.last_decision = None
    session.queued_updater_actions = ()
    session.queued_updater_mode = None
    session.pending_game_over_reset = False
    reset_info = session.environment.get_info()
    session.current_info = reset_info
    session.last_completed_levels = session.completed_levels
    session.frame_buffer = ()
    session.frame_index = 0
    session.last_transition_frame_observations = ()
    session.current = None
    session.next = None
    session.process_turn = False


def finish_run(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> GameRunResult:
    """Apply end-of-run updates, emit stop trace, and return the result."""

    if session.terminal_result is None:
        raise RuntimeError("game-loop session finished without a terminal result")

    result = session.terminal_result
    if result.stop_reason == "game_end":
        apply_general_context_updates(
            contexts=contexts,
            updater_tasks=updater_tasks,
            debug=debug,
            run_id=result.run_id,
            game_id=result.game_id,
            stop_reason=result.stop_reason or "unknown",
            step_count=result.step_count,
            completed_levels=result.completed_levels,
            last_state_name=(
                result.last_state.name if result.last_state is not None else None
            ),
            state_record_ids=result.state_record_ids,
        )
        if state_memory is not None and result.state_record_ids:
            state_memory.update_state_contexts(
                state_id=result.state_record_ids[-1],
                contexts=contexts,
            )
    debug.emit(RunStopped(result))
    return result
