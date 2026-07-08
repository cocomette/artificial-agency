"""Persistence actions for committed game-loop turns."""

from __future__ import annotations

from typing import Any

from face_of_agi.contracts import (
    ActionSpec,
    ContextDocuments,
    FrameTurnContext,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import FrameDecisionRecorded, MStatePersisted
from face_of_agi.orchestration.game_loop.helpers import bounded_action_history
from face_of_agi.orchestration.game_loop.session import GameLoopSession
from face_of_agi.runtime import timing as runtime_timing


def persist_turn(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None,
    contexts: ContextDocuments,
    debug: DebugBus,
) -> None:
    """Persist the completed current frame turn to M."""

    current = require_current(session)
    decision = require_decision(session)
    update_input = require_update_input(session)
    with runtime_timing.span(
        "game_loop.persist_turn_shell",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        persist_turn_shell(
            frame_context=current.to_frame_context(),
            turn_id=current.turn_id,
            decision=decision,
            update_input=update_input,
            state_record_ids=session.state_record_ids,
            state_memory=state_memory,
            contexts=contexts,
            debug=debug,
            agent_context_history=session.agent_context_strategy_snapshot,
            agent_creator_action_history=_agent_creator_action_history(
                session.update_input,
                prior_action_history=session.action_history,
                action_history_window=(
                    session.environment_config.agent_creator.action_history_window
                ),
            ),
            world_model_context=session.world_model_context,
        )


def persist_turn_shell(
    *,
    frame_context: FrameTurnContext,
    turn_id: int,
    decision: Any,
    update_input: UpdaterFrameTransitionInput,
    state_record_ids: list[int],
    state_memory: StateMemory | None,
    contexts: ContextDocuments,
    debug: DebugBus,
    agent_context_history: dict[str, Any] | None = None,
    agent_creator_action_history: tuple[Any, ...] | None = None,
    world_model_context: dict[str, Any] | None = None,
) -> None:
    """Complete the prewritten M row for one frame turn."""

    if state_memory is None or frame_context.current_source_state_id is None:
        return

    state = state_memory.complete_frame_turn_state(
        state_id=frame_context.current_source_state_id,
        turn_id=turn_id,
        control_mode=frame_context.control_mode,
        previous_observation_ref=frame_context.previous_observation_ref,
        recent_action_history=frame_context.recent_action_history,
        chosen_action=decision.final_action,
        contexts=contexts,
        agent_trace=decision.trace,
        turn_metrics=update_input.turn_metrics,
        agent_context_history=agent_context_history,
        agent_creator_action_history=agent_creator_action_history,
        world_model_context=world_model_context,
    )
    state_record_ids.append(state.id)
    debug.emit(MStatePersisted(record_id=state.id, turn_id=turn_id))


def write_frame_trace(
    *,
    debug: DebugBus,
    frame_turn: int,
    frame_context: FrameTurnContext,
    action: ActionSpec,
    trace: Any,
) -> None:
    """Emit the debug trace for one frame decision."""

    debug.emit(
        FrameDecisionRecorded(
            frame_turn=frame_turn,
            frame_context=frame_context,
            action=action,
            trace=trace,
        )
    )


def require_current(session: GameLoopSession):
    if session.current is None:
        raise RuntimeError("game-loop session is missing the current turn")
    return session.current


def require_decision(session: GameLoopSession):
    if session.decision is None:
        raise RuntimeError("game-loop session is missing the frame decision")
    return session.decision


def require_update_input(session: GameLoopSession) -> UpdaterFrameTransitionInput:
    if session.update_input is None:
        raise RuntimeError("game-loop session is missing updater input")
    return session.update_input


def _agent_creator_action_history(
    update_input: UpdaterFrameTransitionInput | None,
    *,
    prior_action_history: tuple[Any, ...],
    action_history_window: int,
) -> tuple[Any, ...] | None:
    if update_input is None:
        return None
    bounded_prior_action_history = bounded_action_history(
        prior_action_history,
        window=action_history_window,
        key="agent_creator.action_history_window",
    )
    if update_input.action_history_entries:
        return (*bounded_prior_action_history, *update_input.action_history_entries)
    if update_input.action_history_entry is not None:
        return (*bounded_prior_action_history, update_input.action_history_entry)
    return bounded_prior_action_history
