"""State-machine step actions for one frame turn."""

from __future__ import annotations

from collections.abc import Callable

from face_of_agi.contracts import (
    ContextDocuments,
    DecisionResult,
    FrameTurnContext,
    PostDecisionPredictions,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models.adapters import OrchestratorAgentModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    EnvironmentStepRecorded,
    FrameTurnStarted,
    PostDecisionPredictionsRecorded,
)
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_context_updates,
)
from face_of_agi.orchestration.game_loop.actions.metrics import (
    effective_trace_cost_seconds,
    turn_metrics,
)
from face_of_agi.orchestration.game_loop.actions.post_decision_predictions import (
    PostDecisionPredictionRunner,
)
from face_of_agi.orchestration.game_loop.helpers import (
    agent_history_window,
    build_action_history_entry,
    decide_frame_turn,
    frame_control_mode,
    unroll_observation,
    validate_decision,
)
from face_of_agi.orchestration.game_loop.persistence import (
    persist_turn,
    write_frame_trace,
)
from face_of_agi.orchestration.game_loop.session import (
    FrameTurnSnapshot,
    GameLoopSession,
)
from face_of_agi.runtime import timing as runtime_timing

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]


def load_frame_buffer_if_needed(session: GameLoopSession) -> None:
    """Normalize the latest environment response into frame turns."""

    if session.frame_buffer and session.frame_index < len(session.frame_buffer):
        return
    session.frame_buffer = unroll_observation(session.latest_environment_observation)
    session.frame_index = 0


def enter_frame_turn(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    state_memory: StateMemory | None,
    tool_runtime_factory: AgentToolRuntimeFactory | None,
    debug: DebugBus,
) -> None:
    """Create the immutable current-frame snapshot and emit turn trace."""

    current_observation = session.frame_buffer[session.frame_index]
    frame_count = len(session.frame_buffer)
    control_mode = frame_control_mode(
        frame_index=session.frame_index,
        frame_count=frame_count,
        real_actions=session.real_actions,
    )
    current_ref = session.current_ref_for(current_observation)
    turn_id = session.frame_turn_count + 1
    source_state = (
        state_memory.prewrite_frame_turn_source(
            run_id=session.config.run_id,
            game_id=session.game_id,
            turn_id=turn_id,
            current_observation=current_observation,
            frame_index=session.frame_index,
            frame_count=frame_count,
            control_mode=control_mode,
            contexts=contexts,
        )
        if state_memory is not None
        else None
    )

    if session.first_observation is None:
        session.first_observation = current_observation
        session.first_observation_ref = current_ref
    if session.first_observation_ref is None:
        raise RuntimeError("frame turn is missing the first observation ref")
    history_anchor_observation, recent_history = agent_history_window(
        session.action_history,
        session.action_history_observations,
        first_observation=session.first_observation,
        window=session.environment_config.action_history_window,
    )

    session.current = FrameTurnSnapshot(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=turn_id,
        observation=current_observation,
        observation_ref=current_ref,
        history_anchor_observation=history_anchor_observation,
        source_state_id=source_state.id if source_state is not None else None,
        frame_index=session.frame_index,
        frame_count=frame_count,
        control_mode=control_mode,
        first_observation_ref=session.first_observation_ref,
        previous_observation_ref=session.previous_observation_ref,
        previous_source_state_id=session.previous_source_state_id,
        recent_action_history=recent_history,
    )
    frame_context = session.current.to_frame_context()
    session.tool_runtime = (
        tool_runtime_factory(
            session.config.run_id,
            session.game_id,
            turn_id,
            frame_context,
        )
        if control_mode.controllable and tool_runtime_factory is not None
        else None
    )
    available_tools = (
        session.tool_runtime.available_tools()
        if session.tool_runtime is not None
        else ()
    )
    debug.emit(
        FrameTurnStarted(
            frame_turn=turn_id,
            frame_context=frame_context,
            lifecycle_state=(
                session.current_info.state
                if session.current_info is not None
                else None
            ),
            completed_levels=session.completed_levels,
            remaining_actions=session.remaining_actions,
            available_tools=available_tools,
        )
    )
    clear_turn_outputs(session)


def decide(
    session: GameLoopSession,
    *,
    agent: OrchestratorAgentModel,
    contexts: ContextDocuments,
    debug: DebugBus,
) -> None:
    """Select or synthesize the current frame-turn action."""

    current = require_current(session)
    frame_context = current.to_frame_context()
    decision, decision_duration_seconds = decide_frame_turn(
        agent=agent,
        contexts=contexts,
        debug=debug,
        frame_context=frame_context,
        tool_runtime=session.tool_runtime,
        history_anchor_observation=current.history_anchor_observation,
        turn_id=current.turn_id,
    )
    session.decision = decision
    session.decision_duration_seconds = decision_duration_seconds
    session.trace_cost_seconds = effective_trace_cost_seconds(
        decision=decision,
        wall_clock_seconds=decision_duration_seconds,
    )
    session.last_decision = decision
    session.frame_turn_count = current.turn_id

    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")
    validate_decision(decision.final_action, control_mode=current.control_mode)
    write_frame_trace(
        debug=debug,
        frame_turn=session.frame_turn_count,
        frame_context=frame_context,
        action=decision.final_action,
        trace=decision.trace,
    )


def run_post_decision_predictions(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    runner: PostDecisionPredictionRunner,
    debug: DebugBus,
) -> None:
    """Run world predictions for the current session turn."""

    current = require_current(session)
    decision = require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")
    with runtime_timing.span(
        "game_loop.post_decision_predictions",
        turn_id=current.turn_id,
        step=current.observation.step,
        controllable=current.control_mode.controllable,
    ):
        session.predictions = _run_post_decision_predictions(
            frame_context=current.to_frame_context(),
            turn_id=current.turn_id,
            current=current,
            decision=decision,
            contexts=contexts,
            runner=runner,
            debug=debug,
        )
    debug.emit(PostDecisionPredictionsRecorded(session.predictions))


def resolve_next_snapshot(session: GameLoopSession, *, debug: DebugBus) -> None:
    """Resolve the observed next frame and assemble transition data."""

    current = require_current(session)
    decision = require_decision(session)
    predictions = require_predictions(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    if current.control_mode.controllable:
        session.real_step_count += 1
        with runtime_timing.span(
            "game_loop.environment_step",
            turn_id=current.turn_id,
            step=current.observation.step,
        ):
            next_observation = session.environment.step(decision.final_action)
        session.remaining_actions -= 1
        session.next_environment_observation = next_observation
        debug.emit(
            EnvironmentStepRecorded(
                action=decision.final_action,
                next_observation=next_observation,
                remaining_actions=session.remaining_actions,
            )
        )
        next_frame_buffer = unroll_observation(next_observation)
        next_frame = next_frame_buffer[0]
        next_frame_index = 0
        next_frame_count = len(next_frame_buffer)
        next_control_mode = None
    else:
        next_frame_index = session.frame_index + 1
        next_frame = session.frame_buffer[next_frame_index]
        next_frame_count = len(session.frame_buffer)
        next_control_mode = frame_control_mode(
            frame_index=next_frame_index,
            frame_count=next_frame_count,
            real_actions=session.real_actions,
        )

    next_ref = session.current_ref_for(next_frame)
    session.turn_metrics = turn_metrics(
        actual_next_observation=next_frame,
        trace_cost_seconds=session.trace_cost_seconds,
        cumulative_time_cost=float(session.real_step_count),
    )
    history_entry = build_action_history_entry(
        frame_context=current.to_frame_context(),
        final_action=decision.final_action,
    )
    next_history_anchor_observation, next_recent_history = agent_history_window(
        (*session.action_history, history_entry),
        (*session.action_history_observations, current.observation),
        first_observation=session.first_observation,
        window=session.environment_config.action_history_window,
    )
    session.next = FrameTurnSnapshot(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id + 1,
        observation=next_frame,
        observation_ref=next_ref,
        history_anchor_observation=next_history_anchor_observation,
        source_state_id=None,
        frame_index=next_frame_index,
        frame_count=next_frame_count,
        control_mode=next_control_mode,
        first_observation_ref=current.first_observation_ref,
        previous_observation_ref=current.observation_ref,
        previous_source_state_id=current.source_state_id,
        recent_action_history=next_recent_history,
    )
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=current.observation_ref,
        actual_next_observation_ref=next_ref,
        decision_trace=decision.trace,
        previous_observation=current.observation,
        actual_next_observation=next_frame,
        post_decision_predictions=predictions,
        turn_metrics=session.turn_metrics,
        submitted_action=(
            decision.final_action if current.control_mode.controllable else None
        ),
        synthetic_none_action=(
            None if current.control_mode.controllable else decision.final_action
        ),
        metadata={"controllable": current.control_mode.controllable},
    )


def run_updaters(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Apply updater P to live contexts for the current transition."""

    current = require_current(session)
    update_input = require_update_input(session)
    with runtime_timing.span(
        "game_loop.apply_context_updates",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        apply_context_updates(
            update_input,
            contexts=contexts,
            updater_tasks=updater_tasks,
            debug=debug,
            state_memory=state_memory,
            frame_context=current.to_frame_context(),
            turn_id=current.turn_id,
        )


def persist(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Persist the current turn through the persistence action module."""

    persist_turn(
        session,
        state_memory=state_memory,
        contexts=contexts,
        debug=debug,
    )


def advance(session: GameLoopSession) -> None:
    """Advance run/session cursors after the current turn is committed."""

    current = require_current(session)
    decision = require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    session.action_history.append(
        build_action_history_entry(
            frame_context=current.to_frame_context(),
            final_action=decision.final_action,
        )
    )
    session.action_history_observations.append(current.observation)
    session.previous_observation_ref = current.observation_ref
    session.previous_source_state_id = current.source_state_id

    if current.control_mode.controllable:
        if session.next_environment_observation is None:
            raise RuntimeError("controllable turn is missing next observation")
        session.latest_environment_observation = session.next_environment_observation
        session.frame_buffer = ()
        session.frame_index = 0
    else:
        session.frame_index += 1

    session.current = None
    session.next = None
    session.tool_runtime = None
    clear_turn_outputs(session)


def clear_turn_outputs(session: GameLoopSession) -> None:
    """Clear transient per-turn outputs before or after a frame turn."""

    session.decision = None
    session.decision_duration_seconds = None
    session.trace_cost_seconds = None
    session.predictions = None
    session.turn_metrics = None
    session.update_input = None
    session.next_environment_observation = None


def require_current(session: GameLoopSession) -> FrameTurnSnapshot:
    if session.current is None:
        raise RuntimeError("game-loop session is missing the current turn")
    return session.current


def require_decision(session: GameLoopSession) -> DecisionResult:
    if session.decision is None:
        raise RuntimeError("game-loop session is missing the frame decision")
    return session.decision


def require_predictions(session: GameLoopSession) -> PostDecisionPredictions:
    if session.predictions is None:
        raise RuntimeError("game-loop session is missing post-decision predictions")
    return session.predictions


def require_update_input(session: GameLoopSession) -> UpdaterFrameTransitionInput:
    if session.update_input is None:
        raise RuntimeError("game-loop session is missing updater input")
    return session.update_input


def _run_post_decision_predictions(
    *,
    frame_context: FrameTurnContext,
    turn_id: int,
    current: FrameTurnSnapshot,
    decision: DecisionResult,
    contexts: ContextDocuments,
    runner: PostDecisionPredictionRunner,
    debug: DebugBus,
) -> PostDecisionPredictions:
    """Run S predictions after X chooses a frame action."""

    predictions = runner.predict(
        current_observation_ref=current.observation_ref,
        current_source_state_id=current.source_state_id,
        current_observation=current.observation,
        final_action=decision.final_action,
        world_context=contexts.world,
    )
    debug.capture_model_inputs(frame_context, turn_id, runner.world_model)
    return predictions
