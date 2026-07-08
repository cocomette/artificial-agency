"""State-machine step actions for one frame turn."""

from __future__ import annotations

import copy
from collections.abc import Callable
from time import perf_counter
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryScoreAdvanceMarker,
    ContextDocuments,
    DecisionResult,
    FrameTurnContext,
    Observation,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models.adapters import (
    AgentContextHistorizerModel,
    OrchestratorAgentModel,
)
from face_of_agi.models.change import ChangeSummaryModel, ChangeSummaryResult
from face_of_agi.models.historizer import AgentContextHistorySummary
from face_of_agi.models.image_inputs import frame_bundle_image_size
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    EnvironmentStepRecorded,
    FrameTurnStarted,
)
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_context_updates,
    summarize_agent_context_history as build_agent_context_history_summary,
)
from face_of_agi.orchestration.game_loop.actions.metrics import (
    effective_trace_cost_seconds,
    turn_metrics,
)
from face_of_agi.orchestration.game_loop.helpers import (
    build_action_history_entry,
    bounded_agent_action_history,
    decide_frame_turn,
    frame_control_mode,
    average_observation_transition_changed_pixel_percent,
    max_observation_transition_changed_pixel_percent,
    model_input_crop_edges,
    observation_visible_changed_pixel_percent,
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

NO_CHANGE_SUMMARY = "no changes"
UNCERTAIN_CHANGE_SUMMARY = (
    "visible pixels changed, but the specific change is uncertain."
)

AgentToolRuntimeFactory = Callable[
    [str, str, int, FrameTurnContext],
    AgentToolRuntime,
]


def load_frame_buffer_if_needed(session: GameLoopSession) -> None:
    """Normalize the latest environment response into frame turns."""

    if session.frame_buffer and session.frame_index < len(session.frame_buffer):
        return
    session.frame_buffer = unroll_observation(
        session.latest_environment_observation,
        animation_keyframe_pixel_threshold=(
            session.environment_config.animation_keyframe_pixel_threshold
        ),
    )
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
    recent_history = bounded_agent_action_history(
        session.action_history,
        window=session.environment_config.agent_action_history_window,
    )

    session.current = FrameTurnSnapshot(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=turn_id,
        observation=current_observation,
        observation_ref=current_ref,
        source_state_id=source_state.id if source_state is not None else None,
        frame_index=session.frame_index,
        frame_count=frame_count,
        control_mode=control_mode,
        first_observation_ref=session.first_observation_ref,
        previous_observation_ref=session.previous_observation_ref,
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
        recent_action_history_available=(
            session.environment_config.agent_action_history_window != 0
        ),
        tool_runtime=session.tool_runtime,
        turn_id=current.turn_id,
        action_suppression_zero_changed_pixel_turns=(
            session.environment_config.action_suppression_zero_changed_pixel_turns
        ),
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


def resolve_next_snapshot(session: GameLoopSession, *, debug: DebugBus) -> None:
    """Resolve the observed next frame and assemble transition data."""

    current = require_current(session)
    decision = require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    if current.control_mode.controllable:
        previous_evidence_observation = snapshot_observation(current.observation)
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
        next_frame_buffer = unroll_observation(
            next_observation,
            animation_keyframe_pixel_threshold=(
                session.environment_config.animation_keyframe_pixel_threshold
            ),
            anchor_frame=previous_evidence_observation.frame,
        )
        session.transition_frame_observations = (
            previous_evidence_observation,
            *next_frame_buffer,
        )
        next_frame = next_frame_buffer[-1]
        session.next_frame_buffer = (next_frame,)
        next_frame_index = 0
        next_frame_count = 1
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
        session.transition_frame_observations = (
            current.observation,
            next_frame,
        )

    next_ref = session.current_ref_for(next_frame)
    session.turn_metrics = turn_metrics(
        actual_next_observation=next_frame,
        trace_cost_seconds=session.trace_cost_seconds,
        cumulative_time_cost=float(session.real_step_count),
    )
    score_advance_marker = build_score_advance_marker(
        previous_score=session.last_observed_cumulative_score,
        new_score=session.turn_metrics.cumulative_score,
    )
    next_recent_history = bounded_agent_action_history(
        session.action_history,
        window=session.environment_config.agent_action_history_window,
    )
    session.next = FrameTurnSnapshot(
        run_id=session.config.run_id,
        game_id=session.game_id,
        turn_id=current.turn_id + 1,
        observation=next_frame,
        observation_ref=next_ref,
        source_state_id=None,
        frame_index=next_frame_index,
        frame_count=next_frame_count,
        control_mode=next_control_mode,
        first_observation_ref=current.first_observation_ref,
        previous_observation_ref=current.observation_ref,
        recent_action_history=next_recent_history,
    )
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=current.observation_ref,
        actual_next_observation_ref=next_ref,
        decision_trace=decision.trace,
        actual_next_observation=next_frame,
        turn_metrics=session.turn_metrics,
        submitted_action=(
            decision.final_action if current.control_mode.controllable else None
        ),
        synthetic_none_action=(
            None if current.control_mode.controllable else decision.final_action
        ),
        action_history_score_advance_marker=score_advance_marker,
        metadata={"controllable": current.control_mode.controllable},
    )


def summarize_change(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel,
    debug: DebugBus,
) -> None:
    """Summarize the observed frame transition for compact action history."""

    result = summarize_change_model(
        session,
        change_model=change_model,
        debug=debug,
    )
    capture_change_summary_inputs(
        session,
        change_model=change_model,
        debug=debug,
    )
    attach_change_summary(
        session,
        result=result,
    )


def summarize_change_model(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel,
    debug: DebugBus | None = None,
) -> ChangeSummaryResult:
    """Run the change-summary model without mutating turn state."""

    current = require_current(session)
    next_snapshot = require_next(session)
    decision = require_decision(session)
    frame_context = current.to_frame_context()
    frame_observations = transition_frame_observations(session)
    view_kwargs = change_model_view_kwargs(
        change_model,
        frame_count=len(frame_observations),
    )
    changed_pixel_percent = observation_visible_changed_pixel_percent(
        frame_observations[0],
        frame_observations[-1],
        **view_kwargs,
    )
    max_transition_changed_pixel_percent = (
        max_observation_transition_changed_pixel_percent(
            frame_observations,
            **view_kwargs,
        )
    )
    animation_avg_changed_pixel_percent = None
    if len(frame_observations) > 2:
        animation_avg_changed_pixel_percent = (
            average_observation_transition_changed_pixel_percent(
                frame_observations,
                **view_kwargs,
            )
        )
    if max_transition_changed_pixel_percent == 0.0:
        return ChangeSummaryResult(
            summary=NO_CHANGE_SUMMARY,
            changed_pixel_percent=changed_pixel_percent,
            change_detected=False,
            metadata={
                "skipped": True,
                "skip_reason": "identical_model_visible_evidence",
                "frame_count": len(frame_observations),
                "max_transition_changed_pixel_percent": (
                    max_transition_changed_pixel_percent
                ),
                "animation_avg_changed_pixel_percent": (
                    animation_avg_changed_pixel_percent
                ),
            },
        )

    started_at = perf_counter()
    with runtime_timing.span(
        "game_loop.change_summary",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        result = change_model.summarize(
            current.observation,
            next_snapshot.observation,
            decision.final_action,
            glossary_actions=frame_context.control_mode.allowed_actions,
            changed_pixel_percent=changed_pixel_percent,
            frame_observations=frame_observations,
            max_transition_changed_pixel_percent=(
                max_transition_changed_pixel_percent
            ),
        )
    if debug is not None:
        emit_model_call_completed(
            debug,
            role="change",
            duration_seconds=perf_counter() - started_at,
        )
    result = ChangeSummaryResult(
        summary=result.summary,
        changed_pixel_percent=result.changed_pixel_percent,
        change_detected=result.change_detected,
        metadata={
            **result.metadata,
            "frame_count": len(frame_observations),
            "max_transition_changed_pixel_percent": (
                max_transition_changed_pixel_percent
            ),
            "animation_avg_changed_pixel_percent": (
                animation_avg_changed_pixel_percent
            ),
        },
    )
    if (
        not result.change_detected
        and max_transition_changed_pixel_percent > 0.0
    ):
        return ChangeSummaryResult(
            summary=UNCERTAIN_CHANGE_SUMMARY,
            changed_pixel_percent=result.changed_pixel_percent,
            change_detected=False,
            metadata={
                **result.metadata,
                "summary_overridden": True,
                "override_reason": "pixel_change_without_model_detected_change",
            },
        )
    return result


def capture_change_summary_inputs(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel,
    debug: DebugBus,
) -> None:
    """Drain debug provider captures for the current change-summary call."""

    current = require_current(session)
    frame_context = current.to_frame_context()
    debug.capture_model_inputs(frame_context, current.turn_id, change_model)


def attach_change_summary(
    session: GameLoopSession,
    *,
    result: ChangeSummaryResult,
) -> None:
    """Attach a completed change summary to the pending updater input."""

    current = require_current(session)
    next_snapshot = require_next(session)
    decision = require_decision(session)
    update_input = require_update_input(session)
    frame_context = current.to_frame_context()
    frame_observations = transition_frame_observations(session)
    retained_animation_frame_count = 0
    skipped_animation_frame_count = None
    animation_avg_changed_pixel_percent = None
    if frame_context.control_mode.controllable:
        retained_animation_frame_count = max(0, len(frame_observations) - 1)
        skipped_animation_frame_count = sum(
            _observation_skipped_animation_frame_count(observation)
            for observation in frame_observations[1:]
        )
        if retained_animation_frame_count > 1:
            animation_avg_changed_pixel_percent = _optional_float_metadata(
                result.metadata.get("animation_avg_changed_pixel_percent")
            )
    update_input.action_history_entry = build_action_history_entry(
        frame_context=frame_context,
        final_action=decision.final_action,
        next_observation=next_snapshot.observation,
        changed_pixel_percent=result.changed_pixel_percent,
        change_summary=result.summary,
        retained_animation_frame_count=retained_animation_frame_count,
        skipped_animation_frame_count=skipped_animation_frame_count,
        animation_avg_changed_pixel_percent=animation_avg_changed_pixel_percent,
    )


def run_updaters(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    agent_context_history: AgentContextHistorySummary,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Apply updater P to live contexts for the current transition."""

    current = require_current(session)
    update_input = require_update_input(session)
    environment_config = session.environment_config
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
            prior_action_history=session.action_history,
            agent_updater_action_history_window=(
                environment_config.agent_updater_action_history_window
            ),
            agent_context_history=agent_context_history,
            action_suppression_zero_changed_pixel_turns=(
                environment_config.action_suppression_zero_changed_pixel_turns
            ),
            updater_stagnation_warning_zero_changed_pixel_turns=(
                environment_config.updater_stagnation_warning_zero_changed_pixel_turns
            ),
            game_last_started_turns_ago=max(
                0,
                current.turn_id - session.game_start_turn_id,
            ),
            score_last_advanced_turns_ago=score_last_advanced_turns_ago(
                session,
                current_turn_id=current.turn_id,
                update_input=update_input,
            ),
            game_start_reason=session.game_start_reason,
            game_restart_count=session.game_restart_count,
            turn_id=current.turn_id,
        )


def summarize_agent_context_history(
    session: GameLoopSession,
    *,
    state_memory: StateMemory | None,
    agent_context_historizer: AgentContextHistorizerModel | None,
    debug: DebugBus | None = None,
) -> AgentContextHistorySummary:
    """Run the agent-context historizer for the current frame turn."""

    current = require_current(session)
    return build_agent_context_history_summary(
        state_memory=state_memory,
        frame_context=current.to_frame_context(),
        historizer=agent_context_historizer,
        context_window=session.environment_config.agent_context_history_window,
        turn_id=current.turn_id,
        debug=debug,
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
    next_snapshot = require_next(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    update_input = require_update_input(session)
    if update_input.action_history_entry is None:
        raise RuntimeError("frame turn is missing a change-summarized history entry")
    session.action_history.append(update_input.action_history_entry)
    if update_input.action_history_score_advance_marker is not None:
        session.action_history.append(update_input.action_history_score_advance_marker)
        session.last_score_advance_turn_id = current.turn_id
    if update_input.turn_metrics.cumulative_score is not None:
        session.last_observed_cumulative_score = float(
            update_input.turn_metrics.cumulative_score
        )
    session.previous_observation_ref = current.observation_ref

    if current.control_mode.controllable:
        if session.next_environment_observation is None:
            raise RuntimeError("controllable turn is missing next observation")
        session.latest_environment_observation = session.next_environment_observation
        if not session.next_frame_buffer:
            raise RuntimeError("controllable turn is missing next frame buffer")
        session.frame_buffer = session.next_frame_buffer
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
    session.turn_metrics = None
    session.update_input = None
    session.candidate_actions = ()
    session.world_predictions = ()
    session.latest_judge_score = None
    session.latest_reward = None
    session.next_environment_observation = None
    session.next_frame_buffer = ()
    session.transition_frame_observations = ()


def build_score_advance_marker(
    *,
    previous_score: float | None,
    new_score: float | None,
) -> ActionHistoryScoreAdvanceMarker | None:
    """Return a score marker for the current-run score state."""

    if new_score is None:
        return None
    current = float(new_score)
    if previous_score is None:
        if current <= 0.0:
            return None
        return ActionHistoryScoreAdvanceMarker(
            previous_score=None,
            new_score=current,
            delta=None,
        )
    previous = float(previous_score)
    if current <= previous:
        return None
    return ActionHistoryScoreAdvanceMarker(
        previous_score=previous,
        new_score=current,
        delta=current - previous,
    )


def score_last_advanced_turns_ago(
    session: GameLoopSession,
    *,
    current_turn_id: int,
    update_input: UpdaterFrameTransitionInput,
) -> int | None:
    """Return the frame-turn distance since the last score marker."""

    if update_input.action_history_score_advance_marker is not None:
        return 0
    if session.last_score_advance_turn_id is None:
        return None
    return max(0, current_turn_id - session.last_score_advance_turn_id)


def require_current(session: GameLoopSession) -> FrameTurnSnapshot:
    if session.current is None:
        raise RuntimeError("game-loop session is missing the current turn")
    return session.current


def require_next(session: GameLoopSession) -> FrameTurnSnapshot:
    if session.next is None:
        raise RuntimeError("game-loop session is missing the next turn")
    return session.next


def require_decision(session: GameLoopSession) -> DecisionResult:
    if session.decision is None:
        raise RuntimeError("game-loop session is missing the frame decision")
    return session.decision


def require_update_input(session: GameLoopSession) -> UpdaterFrameTransitionInput:
    if session.update_input is None:
        raise RuntimeError("game-loop session is missing updater input")
    return session.update_input


def transition_frame_observations(
    session: GameLoopSession,
) -> tuple[Observation, ...]:
    """Return oldest-to-newest frame evidence for the current transition."""

    if session.transition_frame_observations:
        return session.transition_frame_observations
    current = require_current(session)
    next_snapshot = require_next(session)
    return (current.observation, next_snapshot.observation)


def change_model_view_kwargs(
    change_model: ChangeSummaryModel,
    *,
    frame_count: int = 2,
) -> dict[str, Any]:
    """Return image-transform kwargs for deterministic model-visible diffs."""

    config = getattr(change_model, "config", None)
    size = frame_bundle_image_size(
        getattr(config, "input_image_size", None),
        frame_count=frame_count,
    )
    return {
        "frame_scale": getattr(config, "frame_scale", 4),
        "size": size,
        "resample": getattr(config, "input_image_resample", "nearest"),
        "crop_edges": model_input_crop_edges(change_model),
    }


def snapshot_observation(observation: Observation) -> Observation:
    """Return an observation copy insulated from environment frame mutation."""

    if observation.frames:
        frames = tuple(_copy_frame(frame) for frame in observation.frames)
        frame = _copy_frame(observation.frame) if observation.frame is not None else frames[-1]
    else:
        frame = _copy_frame(observation.frame)
        frames = (frame,) if frame is not None else ()
    return Observation(
        id=observation.id,
        step=observation.step,
        frame=frame,
        frames=frames,
        raw_frame_data=observation.raw_frame_data,
        metadata=dict(observation.metadata),
    )


def _copy_frame(frame):
    if frame is None:
        return None
    copy_method = getattr(frame, "copy", None)
    if callable(copy_method):
        try:
            return copy_method()
        except TypeError:
            pass
    return copy.deepcopy(frame)


def _observation_skipped_animation_frame_count(observation: Observation) -> int:
    value = observation.metadata.get("skipped_intermediate_animation_frame_count", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _optional_float_metadata(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not 0 <= numeric <= 100:
        return None
    return numeric


def emit_model_call_completed(
    debug: DebugBus,
    *,
    role: str,
    duration_seconds: float,
) -> None:
    """Emit model timing telemetry when the debug event is available."""

    from face_of_agi.debug.events import ModelCallCompleted

    debug.emit(
        ModelCallCompleted(
            role=role,
            duration_seconds=max(0.0, duration_seconds),
        )
    )
