"""Updater orchestration for frame turns and end-of-run contexts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from time import perf_counter

from face_of_agi.contracts import (
    ActionHistoryItem,
    ContextDocuments,
    FrameTurnContext,
    RoleContext,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models.historizer import (
    AgentContextHistorizerModel,
    AgentContextHistoryInput,
    AgentContextHistorySummary,
)
from face_of_agi.models.updater import (
    AGENT_GAME_CONTEXT_KEYS,
    AgentContextRevisionFeedback,
    AgentGameContextUpdateInput,
    AgentProgressFeedback,
    GeneralKnowledgeUpdateInput,
    UpdaterTaskRegistry,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    ModelCallCompleted,
    UpdaterInputCaptured,
    UpdaterProviderOutputCaptured,
)
from face_of_agi.orchestration.game_loop.helpers import (
    bounded_action_history,
    model_input_crop_edges,
    prompt_action_outcome,
)
from face_of_agi.runtime import timing as runtime_timing


def apply_context_updates(
    update_input: UpdaterFrameTransitionInput,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    prior_action_history: Sequence[ActionHistoryItem],
    agent_updater_action_history_window: int,
    agent_context_history: AgentContextHistorySummary,
    action_suppression_zero_changed_pixel_turns: int,
    updater_stagnation_warning_zero_changed_pixel_turns: int,
    game_last_started_turns_ago: int | None,
    score_last_advanced_turns_ago: int | None,
    game_start_reason: str | None,
    game_restart_count: int,
    turn_id: int,
) -> None:
    """Apply updater P to the live working contexts before persistence."""

    if update_input.actual_next_observation is None:
        raise ValueError("game updaters require the current observation")

    agent_updater_prior_action_history = bounded_action_history(
        prior_action_history,
        window=agent_updater_action_history_window,
        key="agent_updater_action_history_window",
    )

    agent_updater = updater_tasks.require_agent_game_updater()
    agent_action_history = updater_action_history(
        update_input,
        prior_action_history=agent_updater_prior_action_history,
        updater_label="agent",
    )
    stagnation_warning_threshold = (
        updater_stagnation_warning_zero_changed_pixel_turns
        if frame_context.control_mode.controllable
        else 0
    )
    agent_prompt_actions = prompt_action_outcome(
        action_space=frame_context.control_mode.allowed_actions,
        action_history=agent_action_history,
        action_suppression_zero_changed_pixel_turns=(
            action_suppression_zero_changed_pixel_turns
        ),
        updater_stagnation_warning_zero_changed_pixel_turns=(
            stagnation_warning_threshold
        ),
        crop_edges=model_input_crop_edges(agent_updater),
    )
    fresh_game_context_after_reset = _is_fresh_game_over_reset_update(
        game_start_reason=game_start_reason,
        game_last_started_turns_ago=game_last_started_turns_ago,
    )
    previous_agent_context = _agent_previous_context_for_update(
        contexts.agent,
        fresh_game_context=fresh_game_context_after_reset,
    )
    context_revision_feedback = (
        AgentContextRevisionFeedback()
        if fresh_game_context_after_reset
        else _context_revision_feedback(
            current_context=contexts.agent.game,
            state_memory=state_memory,
            frame_context=frame_context,
            lookback=len(agent_updater_prior_action_history),
        )
    )

    agent_update_input = AgentGameContextUpdateInput(
        previous_context=previous_agent_context,
        current_observation=update_input.actual_next_observation,
        allowed_actions=agent_prompt_actions.allowed_actions,
        glossary_actions=frame_context.control_mode.allowed_actions,
        action_history_window=agent_updater_action_history_window,
        context_history=agent_context_history,
        action_history=agent_action_history,
        turn_metrics=AgentProgressFeedback(
            time_cost=update_input.turn_metrics.time_cost,
            cumulative_score=update_input.turn_metrics.cumulative_score,
            game_last_started_turns_ago=game_last_started_turns_ago,
            score_last_advanced_turns_ago=score_last_advanced_turns_ago,
            game_start_reason=game_start_reason,
            game_restart_count=game_restart_count,
        ),
        context_revision_feedback=context_revision_feedback,
        action_outcome_evidence=agent_prompt_actions.evidence,
    )
    debug.emit(UpdaterInputCaptured(role="agent", update_input=agent_update_input))
    started_at = perf_counter()
    with runtime_timing.span("updater.agent_game"):
        try:
            contexts.agent = agent_updater.update_agent_game_context(agent_update_input)
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="updater_agent",
                    duration_seconds=perf_counter() - started_at,
                )
            )
            debug.emit(
                UpdaterProviderOutputCaptured(role="agent", adapter=agent_updater)
            )
    debug.capture_model_inputs(frame_context, turn_id, agent_updater)


def apply_general_context_updates(
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    run_id: str,
    game_id: str,
    stop_reason: str,
    step_count: int,
    completed_levels: int,
    last_state_name: str | None,
    state_record_ids: tuple[int, ...],
) -> None:
    """Apply updater P to general K contexts at the end of a run."""

    common_kwargs = {
        "run_id": run_id,
        "game_id": game_id,
        "stop_reason": stop_reason,
        "step_count": step_count,
        "completed_levels": completed_levels,
        "final_state": last_state_name,
        "state_record_ids": state_record_ids,
        "metadata": {"boundary": "end_of_run"},
    }
    general_updater = updater_tasks.require_general_updater()
    agent_update_input = GeneralKnowledgeUpdateInput(
        role="agent",
        previous_context=contexts.agent,
        **common_kwargs,
    )
    debug.emit(UpdaterInputCaptured(role="agent", update_input=agent_update_input))
    started_at = perf_counter()
    with runtime_timing.span("updater.general_agent"):
        try:
            contexts.agent = general_updater.update_general_knowledge(
                agent_update_input
            )
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="updater_general",
                    duration_seconds=perf_counter() - started_at,
                )
            )
            debug.emit(
                UpdaterProviderOutputCaptured(role="agent", adapter=general_updater)
            )


def updater_action_history(
    update_input: UpdaterFrameTransitionInput,
    *,
    prior_action_history: Sequence[ActionHistoryItem],
    updater_label: str,
) -> tuple[ActionHistoryItem, ...]:
    """Return bounded prior action history plus the current raw transition."""

    if update_input.action_history_entry is None:
        raise ValueError(
            f"{updater_label} game updater requires a current action history entry"
        )
    return (
        *prior_action_history,
        update_input.action_history_entry,
        *(
            (update_input.action_history_score_advance_marker,)
            if update_input.action_history_score_advance_marker is not None
            else ()
        ),
    )


def _is_fresh_game_over_reset_update(
    *,
    game_start_reason: str | None,
    game_last_started_turns_ago: int | None,
) -> bool:
    return (
        game_start_reason == "game_over_reset"
        and game_last_started_turns_ago == 0
    )


def _agent_previous_context_for_update(
    previous_context: RoleContext,
    *,
    fresh_game_context: bool,
) -> RoleContext:
    if not fresh_game_context:
        return previous_context
    return RoleContext(general=previous_context.general, game="")


def _context_revision_feedback(
    *,
    current_context: str,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    lookback: int,
) -> AgentContextRevisionFeedback:
    if state_memory is None:
        return AgentContextRevisionFeedback()
    previous_contexts = state_memory.read_recent_agent_game_contexts(
        game_id=frame_context.game_id,
        run_id=frame_context.run_id,
        before_state_id=frame_context.current_source_state_id,
        limit=lookback,
    )
    return agent_context_revision_feedback(
        current_context=current_context,
        previous_contexts=previous_contexts,
    )


def summarize_agent_context_history(
    *,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    historizer: AgentContextHistorizerModel | None,
    context_window: int,
    turn_id: int,
    debug: DebugBus | None = None,
) -> AgentContextHistorySummary:
    """Summarize prior agent context history without writing debug records."""

    if context_window < 0:
        raise ValueError("agent context history window must be non-negative")
    if context_window == 0 or state_memory is None:
        return AgentContextHistorySummary.not_available()
    previous_contexts = state_memory.read_agent_game_context_history(
        game_id=frame_context.game_id,
        run_id=frame_context.run_id,
        before_state_id=frame_context.current_source_state_id,
        limit=context_window,
    )
    if len(previous_contexts) < 2:
        return AgentContextHistorySummary.not_available()
    if historizer is None:
        raise RuntimeError(
            "agent context history is available but historizer model is not registered"
        )
    history_input = AgentContextHistoryInput(
        game_id=frame_context.game_id,
        context_window=context_window,
        contexts=previous_contexts,
        metadata={
            "run_id": frame_context.run_id,
            "before_state_id": frame_context.current_source_state_id,
        },
    )
    with runtime_timing.span(
        "historizer.agent_context_history",
        turn_id=turn_id,
        context_count=len(previous_contexts),
    ):
        started_at = perf_counter()
        summary = historizer.summarize_agent_context_history(history_input)
    if debug is not None:
        debug.emit(
            ModelCallCompleted(
                role="historizer",
                duration_seconds=perf_counter() - started_at,
            )
        )
    return summary


def agent_context_revision_feedback(
    *,
    current_context: str,
    previous_contexts: Sequence[str],
) -> AgentContextRevisionFeedback:
    """Count field staleness across consecutive prior agent contexts."""

    current = _agent_context_fields(current_context)
    if current is None:
        return AgentContextRevisionFeedback()

    active = {key: True for key in AGENT_GAME_CONTEXT_KEYS}
    counts = {key: 0 for key in AGENT_GAME_CONTEXT_KEYS}
    compared_turns = 0
    for previous_context in previous_contexts:
        previous = _agent_context_fields(previous_context)
        if previous is None:
            break
        compared_turns += 1
        for key in AGENT_GAME_CONTEXT_KEYS:
            if active[key] and previous[key] == current[key]:
                counts[key] += 1
            else:
                active[key] = False
        if not any(active.values()):
            break

    return AgentContextRevisionFeedback(
        compared_turns=compared_turns,
        goals_unchanged_turns=counts["goals"],
        game_mechanics_unchanged_turns=counts["game_mechanics"],
        policy_unchanged_turns=counts["policy"],
        history_unchanged_turns=counts["history"],
        extras_unchanged_turns=counts["extras"],
    )


def _agent_context_fields(text: str) -> dict[str, str] | None:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    if set(loaded) != set(AGENT_GAME_CONTEXT_KEYS):
        return None
    if any(not isinstance(value, str) for value in loaded.values()):
        return None
    return {key: loaded[key] for key in AGENT_GAME_CONTEXT_KEYS}
