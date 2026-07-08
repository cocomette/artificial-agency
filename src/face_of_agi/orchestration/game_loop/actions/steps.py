"""State-machine step actions for one frame turn."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryResetMarker,
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    FrameTurnContext,
    Observation,
    SamePastStateDetection,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.frames import observation_frame_hash
from face_of_agi.models.arc_grid_crop import normalize_arc_grid_crop_edges
from face_of_agi.models.adapters import (
    AgentContextHistorizerModel,
)
from face_of_agi.models.change import ChangeSummaryModel, ChangeSummaryResult
from face_of_agi.models.change import change_summary_elements_text
from face_of_agi.models.historizer import (
    AgentContextHistorySummary,
)
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.models.updater import UpdaterTaskRegistry
from face_of_agi.models.world import AgentContextWorldSummary, AgentWorldModel
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    EnvironmentStepRecorded,
    FrameTurnStarted,
    ModelCallCompleted,
)
from face_of_agi.orchestration.game_loop.actions.context_updates import (
    apply_agent_context_update,
    apply_context_updates,
    agent_context_strategy_snapshot,
    summarize_agent_context_history as build_agent_context_history_summary,
    summarize_agent_world_model as build_agent_world_model,
    updater_action_history,
)
from face_of_agi.orchestration.game_loop.actions.metrics import (
    effective_trace_cost_seconds,
    turn_metrics,
)
from face_of_agi.orchestration.game_loop.helpers import (
    UNCHANGED_FRAME_CHANGE_SUMMARY,
    action_allowed,
    average_consecutive_visible_changed_pixel_count,
    build_action_history_entry,
    bounded_agent_action_history,
    bounded_action_history,
    bundle_frame_observations,
    change_summary_observation_snapshot,
    change_summary_visible_changed_pixel_count,
    change_summary_transition_frame_observations,
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

UNCERTAIN_ACTION_CHANGE_SUMMARY = (
    "This action produced changes but it is uncertain what changed exactly."
)
UNCERTAIN_ANIMATION_CHANGE_SUMMARY = (
    "animation produced changes but it is uncertain what changed exactly."
)


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
    change_model: ChangeSummaryModel | None = None,
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
    frame_hash_crop_edges = _current_frame_hash_crop_edges(change_model)
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
            current_frame_hash=observation_frame_hash(
                current_observation,
                crop_edges=frame_hash_crop_edges,
            ),
            current_frame_hash_crop_edges=frame_hash_crop_edges,
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
    debug: DebugBus,
) -> None:
    """Select or synthesize the current frame-turn action."""

    current = require_current(session)
    frame_context = current.to_frame_context()
    decision, decision_duration_seconds = decide_frame_turn(
        debug=debug,
        frame_context=frame_context,
        queued_actions=session.queued_updater_actions,
        queued_updater_mode=session.queued_updater_mode,
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
    if current.control_mode.controllable:
        session.queued_updater_actions = session.queued_updater_actions[1:]
        if not session.queued_updater_actions:
            session.queued_updater_mode = None
    write_frame_trace(
        debug=debug,
        frame_turn=session.frame_turn_count,
        frame_context=frame_context,
        action=decision.final_action,
        trace=decision.trace,
    )


def resolve_next_snapshot(
    session: GameLoopSession,
    *,
    debug: DebugBus,
    change_model: ChangeSummaryModel | None = None,
) -> None:
    """Resolve the observed next frame and assemble transition data."""

    current = require_current(session)
    decision = require_decision(session)
    if current.control_mode is None:
        raise RuntimeError("current frame snapshot is missing control mode")

    if current.control_mode.controllable:
        session.real_step_count += 1
        change_summary_crop_edges = _change_summary_image_config(change_model)
        previous_transition_observation = change_summary_observation_snapshot(
            current.observation,
            crop_edges=change_summary_crop_edges,
        )
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
        result_frames = bundle_frame_observations(next_observation)
        transition_frame_observations = change_summary_transition_frame_observations(
            previous_observation=previous_transition_observation,
            next_observation=next_observation,
            crop_edges=change_summary_crop_edges,
        )
        next_frame_buffer = (result_frames[-1],)
        session.next_frame_buffer = next_frame_buffer
        session.last_transition_frame_observations = (
            transition_frame_observations
            if len(transition_frame_observations) >= 2
            else ()
        )
        next_frame = result_frames[-1]
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

    next_ref = session.current_ref_for(next_frame)
    session.turn_metrics = turn_metrics(
        actual_next_observation=next_frame,
        trace_cost_seconds=session.trace_cost_seconds,
        cumulative_time_cost=float(session.real_step_count),
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
    if session.update_input is None:
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
            frame_observations=(current.observation, next_frame),
            metadata={"controllable": current.control_mode.controllable},
        )


def prepare_observed_transition(session: GameLoopSession) -> None:
    """Assemble previous-to-current transition input before the next decision."""

    current = require_current(session)
    previous_observation = require_previous_observation(session)
    previous_decision = require_previous_decision(session)
    if current.previous_observation_ref is None:
        raise RuntimeError("observed transition is missing previous observation ref")
    action = previous_decision.final_action
    frame_observations = session.last_transition_frame_observations or (
        previous_observation,
        current.observation,
    )
    session.last_transition_frame_observations = ()
    session.update_input = UpdaterFrameTransitionInput(
        current_observation_ref=current.previous_observation_ref,
        actual_next_observation_ref=current.observation_ref,
        decision_trace=previous_decision.trace,
        actual_next_observation=current.observation,
        turn_metrics=turn_metrics(
            actual_next_observation=current.observation,
            trace_cost_seconds=None,
            cumulative_time_cost=float(session.real_step_count),
        ),
        submitted_action=action if not action.is_none() else None,
        synthetic_none_action=action if action.is_none() else None,
        frame_observations=frame_observations,
        metadata={
            "controllable": not action.is_none(),
            "previous_observation_id": previous_observation.id,
            "animation_frame_count": _animation_frame_count(frame_observations),
        },
    )


def has_observed_transition(session: GameLoopSession) -> bool:
    return session.previous_observation is not None and session.last_decision is not None


def summarize_change(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel,
    debug: DebugBus,
) -> None:
    """Summarize the observed frame transition for compact action history."""

    changed_pixel_count = change_summary_changed_pixel_count(
        session,
        change_model=change_model,
    )
    if changed_pixel_count == 0 and not observed_transition_has_animation_bundle(
        session
    ):
        attach_unchanged_frame_summary(
            session,
            change_model=change_model,
            changed_pixel_count=changed_pixel_count,
        )
        return

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
        change_model=change_model,
        changed_pixel_count=changed_pixel_count,
    )


def summarize_change_model(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel,
    debug: DebugBus,
) -> ChangeSummaryResult:
    """Run the change-summary model without mutating turn state."""

    current = require_current(session)
    previous_observation = _change_previous_observation(session)
    current_observation = _change_current_observation(session)
    decision = _change_decision(session)
    frame_context = current.to_frame_context()
    with runtime_timing.span(
        "game_loop.change_summary",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        started_at = perf_counter()
        try:
            return change_model.summarize(
                previous_observation,
                current_observation,
                decision.final_action,
                glossary_actions=frame_context.control_mode.allowed_actions,
                frame_observations=_change_frame_observations(session),
                previous_change_elements=_previous_change_elements(session),
            )
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="change",
                    duration_seconds=perf_counter() - started_at,
                )
            )


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
    change_model: ChangeSummaryModel | None = None,
    changed_pixel_count: float | None = None,
) -> None:
    """Attach a completed change summary to the pending updater input."""

    current = require_current(session)
    previous_observation = _change_previous_observation(session)
    current_observation = _change_current_observation(session)
    decision = _change_decision(session)
    update_input = require_update_input(session)
    frame_context = current.to_frame_context()
    crop_edges = _change_summary_image_config(change_model)
    action_changed_pixel_count = (
        changed_pixel_count
        if changed_pixel_count is not None
        else change_summary_visible_changed_pixel_count(
            previous_observation,
            current_observation,
            crop_edges=crop_edges,
        )
    )
    action_entry = build_action_history_entry(
        frame_context=frame_context,
        final_action=decision.final_action,
        previous_observation=previous_observation,
        next_observation=current_observation,
        change_summary=_change_summary_for_visible_evidence(
            result,
            changed_pixel_count=action_changed_pixel_count,
            fallback_summary=UNCERTAIN_ACTION_CHANGE_SUMMARY,
        ),
        change_elements=result.elements,
        change_summary_crop_edges=crop_edges,
        supplied_changed_pixel_count=action_changed_pixel_count,
        completed_levels=_completed_levels_after_turn(session),
        action_count=(
            session.real_step_count if not decision.final_action.is_none() else None
        ),
        action_mode=decision.trace.metadata.get("updater_mode"),
        controllable=not decision.final_action.is_none(),
    )
    entries = [action_entry]
    animation_frame_count = _animation_frame_count(update_input.frame_observations)
    if animation_frame_count > 1:
        avg_changed_pixel_count = average_consecutive_visible_changed_pixel_count(
            update_input.frame_observations,
            crop_edges=crop_edges,
        )
        entries.append(
            build_action_history_entry(
                frame_context=frame_context,
                final_action=ActionSpec.none(),
                previous_observation=previous_observation,
                next_observation=current_observation,
                change_summary=_change_summary_for_visible_evidence(
                    result,
                    changed_pixel_count=avg_changed_pixel_count,
                    fallback_summary=UNCERTAIN_ANIMATION_CHANGE_SUMMARY,
                ),
                change_elements=result.elements,
                change_summary_crop_edges=crop_edges,
                supplied_changed_pixel_count=avg_changed_pixel_count,
                completed_levels=_completed_levels_after_turn(session),
                action_count=None,
                action_mode=None,
                controllable=False,
                animation_frame_count=animation_frame_count,
                avg_changed_pixel_count=avg_changed_pixel_count,
            )
        )
    update_input.action_history_entry = action_entry
    update_input.action_history_entries = tuple(entries)


def _change_summary_for_visible_evidence(
    result: ChangeSummaryResult,
    *,
    changed_pixel_count: float,
    fallback_summary: str,
) -> str:
    if result.change_detected or changed_pixel_count == 0:
        summary = change_summary_elements_text(result.elements)
        if summary:
            return summary
        if changed_pixel_count == 0:
            return UNCHANGED_FRAME_CHANGE_SUMMARY
        return ""
    return fallback_summary


def attach_unchanged_frame_summary(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel | None,
    changed_pixel_count: float = 0.0,
) -> None:
    """Attach a local identical-frame summary without calling the model."""

    attach_change_summary(
        session,
        result=ChangeSummaryResult(
            elements=(),
            change_detected=False,
            metadata={"skipped_change_summary": True},
        ),
        change_model=change_model,
        changed_pixel_count=changed_pixel_count,
    )


def change_summary_changed_pixel_count(
    session: GameLoopSession,
    *,
    change_model: ChangeSummaryModel | None,
) -> float:
    """Return the current transition's visible changed-pixel percentage."""

    update_input = session.update_input
    if update_input is not None and len(update_input.frame_observations) >= 2:
        previous_observation = update_input.frame_observations[0]
        current_observation = update_input.frame_observations[-1]
    else:
        previous_observation = _change_previous_observation(session)
        current_observation = _change_current_observation(session)
    crop_edges = _change_summary_image_config(change_model)
    return change_summary_visible_changed_pixel_count(
        previous_observation,
        current_observation,
        crop_edges=crop_edges,
    )


def observed_transition_has_animation_bundle(session: GameLoopSession) -> bool:
    """Return whether the current observed transition has bundled animation."""

    update_input = require_update_input(session)
    return _animation_frame_count(update_input.frame_observations) > 1


def _change_summary_image_config(
    change_model: ChangeSummaryModel | None,
) -> object | None:
    """Return image rendering settings used by the change-summary role."""

    config = getattr(change_model, "config", None)
    if config is None:
        return None
    return getattr(config, "input_image_crop_arc_grid_edges", None)


def _current_frame_hash_crop_edges(
    change_model: ChangeSummaryModel | None,
) -> tuple[int, int, int, int]:
    return normalize_arc_grid_crop_edges(_change_summary_image_config(change_model))


def run_updaters(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    world_model: AgentWorldModel | None,
    agent_context_historizer: AgentContextHistorizerModel | None,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory | None,
    debug: DebugBus,
) -> None:
    """Apply updater P to live contexts for the current transition."""

    current = require_current(session)
    update_input = require_update_input(session)
    previous_observation = require_previous_observation(session)
    environment_config = session.environment_config
    frame_context = current.to_frame_context()
    world_prior_action_history = bounded_action_history(
        session.action_history,
        window=environment_config.world_action_history_window,
        key="world_action_history_window",
    )
    world_action_history = updater_action_history(
        update_input,
        prior_action_history=world_prior_action_history,
        updater_label="world",
    )
    agent_world_model = build_agent_world_model(
        state_memory=state_memory,
        frame_context=frame_context,
        world_model=world_model,
        current_observation=update_input.actual_next_observation,
        action_history=world_action_history,
        allowed_actions=frame_context.control_mode.allowed_actions,
        turn_id=current.turn_id,
        debug=debug,
    )
    debug.capture_model_inputs(frame_context, current.turn_id, world_model)
    historizer_prior_action_history = bounded_action_history(
        session.action_history,
        window=environment_config.historizer_action_history_window,
        key="historizer_action_history_window",
    )
    historizer_action_history = updater_action_history(
        update_input,
        prior_action_history=historizer_prior_action_history,
        updater_label="historizer",
    )
    agent_context_history = build_agent_context_history_summary(
        state_memory=state_memory,
        frame_context=frame_context,
        historizer=agent_context_historizer,
        context_window=environment_config.agent_context_history_window,
        previous_observation=previous_observation,
        current_observation=update_input.actual_next_observation,
        action_history=historizer_action_history,
        allowed_actions=frame_context.control_mode.allowed_actions,
        current_world_model=agent_world_model,
        turn_id=current.turn_id,
        debug=debug,
    )
    _store_world_model_context(session, agent_context_history)
    _store_agent_context_evolution_snapshot(session, agent_context_history)
    debug.capture_model_inputs(
        frame_context,
        current.turn_id,
        agent_context_historizer,
    )
    with runtime_timing.span(
        "game_loop.apply_context_updates",
        turn_id=current.turn_id,
        step=current.observation.step,
    ):
        next_actions, updater_mode = apply_context_updates(
            update_input,
            contexts=contexts,
            updater_tasks=updater_tasks,
            debug=debug,
            frame_context=frame_context,
            prior_action_history=session.action_history,
            historizer_action_history=historizer_action_history,
            historizer_action_history_window=(
                environment_config.historizer_action_history_window
            ),
            probing_action_history_window=(
                environment_config.probing_action_history_window
            ),
            policy_action_history_window=(
                environment_config.policy_action_history_window
            ),
            agent_context_history=agent_context_history,
            game_last_started_turns_ago=max(
                0,
                current.turn_id - session.game_start_turn_id,
            ),
            game_start_reason=session.game_start_reason,
            probing_actions_window=environment_config.probing_actions_window,
            policy_actions_window=environment_config.policy_actions_window,
            probing_mode_cap_ratio=environment_config.probing_mode_cap_ratio,
            turn_id=current.turn_id,
            previous_level_solution_method=_latest_level_solution_method(
                state_memory,
                run_id=session.config.run_id,
                game_id=session.game_id,
            ),
            same_past_state_detections=_same_past_state_detections(
                state_memory,
                frame_context=frame_context,
            ),
        )
    _store_agent_context_strategy_snapshot(session, contexts)
    session.queued_updater_actions = next_actions
    session.queued_updater_mode = updater_mode


def should_run_updaters(session: GameLoopSession) -> bool:
    """Return whether this transition needs fresh updater context/actions."""

    current = require_current(session)
    if current.control_mode is None or not current.control_mode.controllable:
        return False
    if session.pending_game_over_reset:
        return True
    if not session.queued_updater_actions:
        return True
    return not action_allowed(
        session.queued_updater_actions[0],
        control_mode=current.control_mode,
    )


def clear_queued_actions_after_net_noop_transition(session: GameLoopSession) -> None:
    """Drop queued actions when the latest real action made no net frame change."""

    update_input = require_update_input(session)
    entry = update_input.action_history_entry
    if entry is None:
        return
    if not entry.controllable or entry.action.is_none():
        return
    if entry.changed_pixel_count != 0:
        return
    session.queued_updater_actions = ()
    session.queued_updater_mode = None


def attach_game_over_reset_decision(session: GameLoopSession) -> None:
    """Attach the synthetic decision used to persist a modeled GAME_OVER turn."""

    current = require_current(session)
    final_action = ActionSpec.none()
    session.decision = DecisionResult(
        final_action=final_action,
        trace=AgentTrace(
            step=current.observation.step,
            first_observation_ref=current.first_observation_ref,
            current_observation_ref=current.observation_ref,
            final_action=final_action,
            reasoning_summary="game-over transition modeled before reset",
            metadata={
                "decision_source": "orchestration_game_over_reset",
                "agent_x_called": False,
            },
        ),
    )
    session.decision_duration_seconds = 0.0
    session.trace_cost_seconds = 0.0
    session.last_decision = session.decision
    session.frame_turn_count = current.turn_id


def record_action_history(session: GameLoopSession) -> None:
    """Append the current transition's action-history entries to the session."""

    update_input = require_update_input(session)
    if update_input.action_history_entries:
        session.action_history.extend(update_input.action_history_entries)
    elif update_input.action_history_entry is not None:
        session.action_history.append(update_input.action_history_entry)


def _completed_levels_after_turn(session: GameLoopSession) -> int | None:
    """Return cumulative completed levels after the current transition."""

    metrics = session.turn_metrics
    if metrics is not None and metrics.cumulative_score is not None:
        return int(metrics.cumulative_score)
    return int(session.completed_levels)


def bootstrap_agent_updater_decision(
    session: GameLoopSession,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    state_memory: StateMemory | None = None,
    debug: DebugBus,
) -> None:
    """Produce the first updater action before any transition exists."""

    current = require_current(session)
    frame_context = current.to_frame_context()
    if not frame_context.control_mode.controllable:
        return
    if session.queued_updater_actions:
        return
    if _has_prior_controllable_action(session):
        return

    environment_config = session.environment_config
    agent_action_history = bounded_action_history(
        session.action_history,
        window=environment_config.probing_action_history_window,
        key="probing_action_history_window",
    )
    agent_context_history = _initial_probing_context_history()
    _store_agent_context_evolution_snapshot(session, agent_context_history)

    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=updater_tasks,
        debug=debug,
        frame_context=frame_context,
        current_observation=current.observation,
        action_history=agent_action_history,
        allowed_action_source=frame_context.control_mode.allowed_actions,
        agent_context_history=agent_context_history,
        turn_id=current.turn_id,
        probing_actions_window=environment_config.probing_actions_window,
        policy_actions_window=environment_config.policy_actions_window,
        probing_mode_cap_ratio=environment_config.probing_mode_cap_ratio,
        previous_level_solution_method=_latest_level_solution_method(
            state_memory,
            run_id=session.config.run_id,
            game_id=session.game_id,
        ),
        same_past_state_detections=_same_past_state_detections(
            state_memory,
            frame_context=frame_context,
        ),
        fresh_game_context_after_reset=(
            session.game_start_reason == "game_over_reset"
            and current.turn_id == session.game_start_turn_id
        ),
    )
    _store_agent_context_strategy_snapshot(session, contexts)
    session.queued_updater_actions = result.next_actions
    session.queued_updater_mode = result.updater_mode


def _has_prior_controllable_action(session: GameLoopSession) -> bool:
    for item in reversed(session.action_history):
        if isinstance(item, ActionHistoryResetMarker):
            return False
        if isinstance(item, ActionHistoryEntry) and item.controllable:
            return True
    return False


def _initial_probing_context_history() -> AgentContextHistorySummary:
    return AgentContextHistorySummary(
        world_description="",
        action_effects={},
        updater_mode="probing",
        probing_evolution="",
        policy_evolution="",
        metadata={"available": False, "bootstrap": True},
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

    record_action_history(session)
    session.previous_observation_ref = current.observation_ref
    session.previous_observation = current.observation

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
    session.next_environment_observation = None
    session.world_model_context = None
    session.agent_context_strategy_snapshot = None
    session.agent_context_evolution_snapshot = None
    session.next_frame_buffer = ()


def require_previous_observation(session: GameLoopSession) -> Observation:
    if session.previous_observation is None:
        raise RuntimeError("game-loop session is missing the previous observation")
    return session.previous_observation


def require_previous_decision(session: GameLoopSession) -> DecisionResult:
    if session.last_decision is None:
        raise RuntimeError("game-loop session is missing the previous decision")
    return session.last_decision


def _change_previous_observation(session: GameLoopSession) -> Observation:
    if has_observed_transition(session):
        return require_previous_observation(session)
    return require_current(session).observation


def _change_current_observation(session: GameLoopSession) -> Observation:
    if has_observed_transition(session):
        return require_current(session).observation
    return require_next(session).observation


def _change_decision(session: GameLoopSession) -> DecisionResult:
    if has_observed_transition(session):
        return require_previous_decision(session)
    return require_decision(session)


def _change_frame_observations(
    session: GameLoopSession,
) -> tuple[Observation, ...] | None:
    frame_observations = require_update_input(session).frame_observations
    if len(frame_observations) > 2:
        return frame_observations
    return None


def _previous_change_elements(
    session: GameLoopSession,
) -> tuple[Any, ...]:
    for item in reversed(session.action_history):
        if isinstance(item, ActionHistoryResetMarker):
            return ()
        if isinstance(item, ActionHistoryEntry) and item.change_elements:
            return item.change_elements
    return ()


def _animation_frame_count(observations: tuple[Observation, ...]) -> int:
    if len(observations) <= 2:
        return 0
    return len(observations) - 1


def _store_agent_context_strategy_snapshot(
    session: GameLoopSession,
    contexts: ContextDocuments,
) -> None:
    session.agent_context_strategy_snapshot = agent_context_strategy_snapshot(contexts)


def _store_agent_context_evolution_snapshot(
    session: GameLoopSession,
    summary: AgentContextHistorySummary,
) -> None:
    session.agent_context_evolution_snapshot = {
        "probing_evolution": summary.probing_evolution,
        "policy_evolution": summary.policy_evolution,
    }


def _store_world_model_context(
    session: GameLoopSession,
    summary: AgentContextHistorySummary | AgentContextWorldSummary,
) -> None:
    world_description = getattr(summary, "world_description")
    action_effects = getattr(summary, "action_effects")
    special_events = getattr(summary, "special_events", "")
    session.world_model_context = {
        "world_description": world_description,
        "special_events": special_events,
        "action_effects": dict(action_effects),
    }


def _latest_level_solution_method(
    state_memory: StateMemory | None,
    *,
    run_id: str,
    game_id: str,
) -> str:
    if state_memory is None:
        return ""
    summary = state_memory.read_latest_level_solution_summary(
        run_id=run_id,
        game_id=game_id,
    )
    if summary is None:
        return ""
    return summary.solution_method


def _same_past_state_detections(
    state_memory: StateMemory | None,
    *,
    frame_context: FrameTurnContext,
) -> tuple[SamePastStateDetection, ...]:
    if state_memory is None or frame_context.current_source_state_id is None:
        return ()
    source = state_memory.read_state_source(frame_context.current_source_state_id)
    if source is None:
        raise RuntimeError("current source M row is missing for same-state lookup")
    frame_hash = source.metadata.get("current_frame_hash")
    if not isinstance(frame_hash, str) or not frame_hash:
        raise RuntimeError("current source M row is missing current_frame_hash")
    return state_memory.read_same_past_state_detections(
        game_id=frame_context.game_id,
        run_id=frame_context.run_id,
        before_state_id=frame_context.current_source_state_id,
        current_frame_hash=frame_hash,
    )


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
