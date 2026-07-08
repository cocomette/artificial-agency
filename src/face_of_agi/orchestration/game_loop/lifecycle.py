"""Run lifecycle actions and decisions for the game-loop state machine."""

from __future__ import annotations

import time

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
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import RunStarted, RunStopped
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_general_context_updates,
)
from face_of_agi.orchestration.game_loop.session import GameLoopSession
from face_of_agi.runtime.source_metadata import (
    RUNTIME_STARTUP_METADATA_KIND,
    build_runtime_source_metadata,
)

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
    if state_memory is not None:
        state_memory.write_run_metadata(
            run_id=config.run_id,
            game_id=selected_game_id,
            kind=RUNTIME_STARTUP_METADATA_KIND,
            metadata=build_runtime_source_metadata(),
        )
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


def check_lifecycle(session: GameLoopSession) -> None:
    """Handle lifecycle states before processing a frame turn."""

    info = session.environment.get_info()
    session.current_info = info
    state = info.state

    if state == GameState.WIN:
        stop_session(
            session,
            stop_reason="game_end",
            completed_levels=info.levels_completed,
            last_state=state,
        )
        return

    completed_levels = max(session.completed_levels, info.levels_completed)
    if completed_levels > session.completed_levels:
        session.completed_levels = completed_levels
        session.last_completed_levels = completed_levels
        session.remaining_actions = session.environment_config.max_actions_per_level

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
        reset_after_game_over(session)
        return

    session.real_actions = tuple(info.available_actions) or tuple(
        session.environment.get_action_space()
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
    reset_info = session.environment.get_info()
    session.current_info = reset_info
    session.last_completed_levels = session.completed_levels
    session.frame_buffer = ()
    session.frame_index = 0
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
