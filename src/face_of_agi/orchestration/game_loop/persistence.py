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
        post_decision_predictions=update_input.post_decision_predictions,
        turn_metrics=update_input.turn_metrics,
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
