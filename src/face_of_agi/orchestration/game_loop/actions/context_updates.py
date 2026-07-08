"""Updater orchestration for frame turns and end-of-run contexts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionSpec,
    ContextDocuments,
    FrameTurnContext,
    Observation,
    RoleContext,
    SamePastStateDetection,
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
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    AgentUpdaterMode,
    GeneralKnowledgeUpdateInput,
    UpdaterTaskRegistry,
)
from face_of_agi.models.world import (
    AgentContextWorldSummary,
    AgentWorldModel,
    AgentWorldModelInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    ModelCallCompleted,
    UpdaterInputCaptured,
    UpdaterProviderOutputCaptured,
)
from face_of_agi.orchestration.game_loop.helpers import (
    bounded_action_history,
)
from face_of_agi.runtime import timing as runtime_timing

DEFAULT_PROBING_MODE_CAP_RATIO = 0.35


def apply_context_updates(
    update_input: UpdaterFrameTransitionInput,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    prior_action_history: Sequence[ActionHistoryItem],
    historizer_action_history: Sequence[ActionHistoryItem],
    historizer_action_history_window: int,
    probing_action_history_window: int,
    policy_action_history_window: int,
    agent_context_history: AgentContextHistorySummary,
    game_last_started_turns_ago: int | None,
    game_start_reason: str | None,
    probing_actions_window: int,
    policy_actions_window: int,
    probing_mode_cap_ratio: float,
    turn_id: int,
    previous_level_solution_method: str = "",
    same_past_state_detections: tuple[SamePastStateDetection, ...] = (),
) -> tuple[tuple[ActionSpec, ...], AgentUpdaterMode]:
    """Apply updater P to the live working contexts before persistence."""

    if update_input.actual_next_observation is None:
        raise ValueError("game updaters require the current observation")
    if update_input.action_history_entry is None:
        raise ValueError("game updaters require a current action history entry")

    agent_context_history = _apply_probing_mode_cap(
        agent_context_history,
        action_history=historizer_action_history,
        probing_actions_window=probing_actions_window,
        historizer_action_history_window=historizer_action_history_window,
        probing_mode_cap_ratio=probing_mode_cap_ratio,
    )
    updater_action_history_window = (
        probing_action_history_window
        if agent_context_history.updater_mode == "probing"
        else policy_action_history_window
    )
    bounded_prior_action_history = bounded_action_history(
        prior_action_history,
        window=updater_action_history_window,
        key=f"{agent_context_history.updater_mode}_action_history_window",
    )
    agent_action_history = updater_action_history(
        update_input,
        prior_action_history=bounded_prior_action_history,
        updater_label="agent",
    )
    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=updater_tasks,
        debug=debug,
        frame_context=frame_context,
        current_observation=update_input.actual_next_observation,
        action_history=agent_action_history,
        allowed_action_source=frame_context.control_mode.allowed_actions,
        agent_context_history=agent_context_history,
        previous_level_solution_method=previous_level_solution_method,
        same_past_state_detections=same_past_state_detections,
        turn_id=turn_id,
        probing_actions_window=probing_actions_window,
        policy_actions_window=policy_actions_window,
        probing_mode_cap_ratio=probing_mode_cap_ratio,
        fresh_game_context_after_reset=_is_fresh_game_over_reset_update(
            game_start_reason=game_start_reason,
            game_last_started_turns_ago=game_last_started_turns_ago,
        ),
    )
    return result.next_actions, result.updater_mode


def apply_agent_context_update(
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    current_observation: Observation,
    action_history: Sequence[ActionHistoryItem],
    allowed_action_source: Sequence[ActionSpec],
    agent_context_history: AgentContextHistorySummary,
    turn_id: int,
    probing_actions_window: int = 1,
    policy_actions_window: int = 1,
    historizer_action_history_window: int | None = None,
    probing_mode_cap_ratio: float = DEFAULT_PROBING_MODE_CAP_RATIO,
    previous_level_solution_method: str = "",
    same_past_state_detections: tuple[SamePastStateDetection, ...] = (),
    fresh_game_context_after_reset: bool = False,
) -> AgentGameContextUpdateResult:
    """Run exactly one mode-specific agent updater and merge its field."""

    if agent_context_history.updater_mode not in {"probing", "policy"}:
        raise RuntimeError(
            "agent context historizer returned invalid updater_mode: "
            f"{agent_context_history.updater_mode!r}"
        )
    agent_context_history = _apply_probing_mode_cap(
        agent_context_history,
        action_history=action_history,
        probing_actions_window=probing_actions_window,
        historizer_action_history_window=historizer_action_history_window,
        probing_mode_cap_ratio=probing_mode_cap_ratio,
    )
    current_fields = _agent_context_fields_for_update(
        contexts.agent.game,
        fresh_game_context=fresh_game_context_after_reset,
    )
    updater_mode = agent_context_history.updater_mode
    updated_key = _agent_context_key_for_mode(updater_mode)
    previous_agent_context = RoleContext(
        general=contexts.agent.general,
        game=_dump_agent_context_fields(current_fields),
    )
    agent_update_input = AgentGameContextUpdateInput(
        previous_context=previous_agent_context,
        current_observation=current_observation,
        allowed_actions=tuple(allowed_action_source),
        glossary_actions=tuple(allowed_action_source),
        context_history=agent_context_history,
        same_past_state_detections=same_past_state_detections,
        previous_level_solution_method=previous_level_solution_method,
        action_history=tuple(action_history),
        actions_window=(
            probing_actions_window
            if updater_mode == "probing"
            else policy_actions_window
        ),
    )
    task_label = f"agent_{updater_mode}"
    debug.emit(UpdaterInputCaptured(role=task_label, update_input=agent_update_input))
    if updater_mode == "probing":
        agent_updater = updater_tasks.require_agent_probing_updater()
        span_name = "updater.agent_probing"
        update = agent_updater.update_agent_probing_context
    else:
        agent_updater = updater_tasks.require_agent_policy_updater()
        span_name = "updater.agent_policy"
        update = agent_updater.update_agent_policy_context
    with runtime_timing.span(span_name):
        started_at = perf_counter()
        try:
            result = update(agent_update_input)
        finally:
            debug.emit(
                ModelCallCompleted(
                    role=task_label,
                    duration_seconds=perf_counter() - started_at,
                )
            )
            debug.emit(
                UpdaterProviderOutputCaptured(role=task_label, adapter=agent_updater)
            )
    debug.capture_model_inputs(frame_context, turn_id, agent_updater)
    if result.updater_mode != updater_mode:
        raise RuntimeError(
            "agent updater returned mismatched updater_mode: "
            f"{result.updater_mode!r} for selected {updater_mode!r}"
        )
    if len(result.next_actions) != agent_update_input.actions_window:
        raise RuntimeError(
            f"{task_label} updater returned {len(result.next_actions)} actions, "
            f"expected exactly {agent_update_input.actions_window}"
        )
    updated_fields = _agent_context_fields(result.context, expected_keys=(updated_key,))
    if set(updated_fields) != {updated_key}:
        raise RuntimeError(
            f"{task_label} updater must return only {updated_key!r}"
        )
    merged_fields = dict(current_fields)
    merged_fields[updated_key] = updated_fields[updated_key]
    contexts.agent = RoleContext(
        general=contexts.agent.general,
        game=_dump_agent_context_fields(merged_fields),
    )
    return result


def _apply_probing_mode_cap(
    agent_context_history: AgentContextHistorySummary,
    *,
    action_history: Sequence[ActionHistoryItem],
    probing_actions_window: int,
    historizer_action_history_window: int | None,
    probing_mode_cap_ratio: float,
) -> AgentContextHistorySummary:
    if agent_context_history.updater_mode != "probing":
        return agent_context_history
    if (
        historizer_action_history_window is None
        or historizer_action_history_window <= 0
    ):
        return agent_context_history
    bounded_history = bounded_action_history(
        action_history,
        window=historizer_action_history_window,
        key="historizer_action_history_window",
    )
    probing_mode_count = sum(
        1
        for item in bounded_history
        if (
            isinstance(item, ActionHistoryEntry)
            and item.controllable
            and item.action_mode == "probing"
        )
    )
    probing_ratio = (
        probing_actions_window + probing_mode_count
    ) / historizer_action_history_window
    if probing_ratio <= probing_mode_cap_ratio:
        return agent_context_history
    return replace(
        agent_context_history,
        updater_mode="policy",
        metadata={
            **agent_context_history.metadata,
            "updater_mode_override": {
                "from": "probing",
                "to": "policy",
                "reason": "probing_mode_cap",
                "probing_actions_window": probing_actions_window,
                "probing_mode_count": probing_mode_count,
                "historizer_action_history_window": (
                    historizer_action_history_window
                ),
                "probing_ratio": probing_ratio,
                "cap_ratio": probing_mode_cap_ratio,
            },
        },
    )


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
    with runtime_timing.span("updater.general_agent"):
        try:
            contexts.agent = general_updater.update_general_knowledge(
                agent_update_input
            )
        finally:
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
    if update_input.action_history_entries:
        return (
            *prior_action_history,
            *update_input.action_history_entries,
        )
    return (
        *prior_action_history,
        update_input.action_history_entry,
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


def summarize_agent_context_history(
    *,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    historizer: AgentContextHistorizerModel | None,
    context_window: int,
    previous_observation: Observation | None,
    current_observation: Observation,
    action_history: Sequence[ActionHistoryItem],
    allowed_actions: Sequence[ActionSpec],
    current_world_model: AgentContextWorldSummary,
    turn_id: int,
    debug: DebugBus,
) -> AgentContextHistorySummary:
    """Summarize prior agent context history without writing debug records."""

    if context_window < 0:
        raise ValueError("agent context history window must be non-negative")
    if historizer is None:
        raise RuntimeError("agent context historizer model is not registered")
    if context_window == 0 or state_memory is None:
        previous_summaries: tuple[str, ...] = ()
    else:
        previous_summaries = state_memory.read_agent_context_history(
            game_id=frame_context.game_id,
            run_id=frame_context.run_id,
            before_state_id=frame_context.current_source_state_id,
            limit=context_window,
        )
    previous_world_model = (
        ""
        if state_memory is None
        else state_memory.read_world_model_context_before(
            game_id=frame_context.game_id,
            before_state_id=frame_context.current_source_state_id,
        )
    )
    history_input = AgentContextHistoryInput(
        game_id=frame_context.game_id,
        context_window=context_window,
        strategy_history=previous_summaries,
        current_world_model=current_world_model,
        previous_world_model=previous_world_model,
        previous_observation=previous_observation,
        current_observation=current_observation,
        action_history=tuple(action_history),
        allowed_actions=tuple(allowed_actions),
        metadata={
            "run_id": frame_context.run_id,
            "before_state_id": frame_context.current_source_state_id,
        },
    )
    with runtime_timing.span(
        "historizer.agent_context_history",
        turn_id=turn_id,
        context_count=len(previous_summaries),
    ):
        started_at = perf_counter()
        try:
            summary = historizer.summarize_agent_context_history(history_input)
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="historizer",
                    duration_seconds=perf_counter() - started_at,
                )
            )
    return summary


def summarize_agent_world_model(
    *,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    world_model: AgentWorldModel | None,
    current_observation: Observation,
    action_history: Sequence[ActionHistoryItem],
    allowed_actions: Sequence[ActionSpec],
    turn_id: int,
    debug: DebugBus,
) -> AgentContextWorldSummary:
    """Summarize world-model context without selecting updater mode."""

    if world_model is None:
        raise RuntimeError("world model is not registered")
    previous_world_model = (
        ""
        if state_memory is None
        else state_memory.read_world_model_context_before(
            game_id=frame_context.game_id,
            before_state_id=frame_context.current_source_state_id,
        )
    )
    world_input = AgentWorldModelInput(
        game_id=frame_context.game_id,
        previous_world_model=previous_world_model,
        current_observation=current_observation,
        action_history=tuple(action_history),
        allowed_actions=tuple(allowed_actions),
        metadata={
            "run_id": frame_context.run_id,
            "before_state_id": frame_context.current_source_state_id,
        },
    )
    with runtime_timing.span(
        "world_model.agent_context",
        turn_id=turn_id,
    ):
        started_at = perf_counter()
        try:
            return world_model.summarize_agent_world_model(world_input)
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="world",
                    duration_seconds=perf_counter() - started_at,
                )
            )


def _agent_context_fields_for_update(
    text: str,
    *,
    fresh_game_context: bool,
) -> dict[str, Any]:
    if fresh_game_context or not text.strip():
        return _blank_agent_context_fields()
    return _agent_context_fields(text)


def agent_context_strategy_snapshot(contexts: ContextDocuments) -> dict[str, str]:
    """Return the current two-field agent game context for history storage."""

    return _agent_context_fields(contexts.agent.game)


def _agent_context_fields(
    text: str,
    *,
    expected_keys: Sequence[str] = AGENT_GAME_CONTEXT_KEYS,
) -> dict[str, Any]:
    expected = ", ".join(expected_keys)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("agent game context must be two-field JSON") from None
    if not isinstance(loaded, dict):
        raise RuntimeError("agent game context must be a JSON object")
    if set(loaded) != set(expected_keys):
        raise RuntimeError(f"agent game context must contain exactly: {expected}")
    _validate_agent_context_field_values(loaded, expected_keys=expected_keys)
    return {key: loaded[key] for key in expected_keys}


def _validate_agent_context_field_values(
    fields: dict[str, Any],
    *,
    expected_keys: Sequence[str] = AGENT_GAME_CONTEXT_KEYS,
) -> None:
    for key in expected_keys:
        if not isinstance(fields.get(key), str):
            raise RuntimeError(f"agent game context {key} must be a string")


def _blank_agent_context_fields() -> dict[str, Any]:
    return {
        "probing_strategy": "",
        "policy_strategy": "",
    }


def _dump_agent_context_fields(fields: dict[str, Any]) -> str:
    return json.dumps(
        {key: fields[key] for key in fields},
        indent=2,
        ensure_ascii=False,
    )


def _agent_context_key_for_mode(mode: AgentUpdaterMode) -> str:
    if mode == "probing":
        return "probing_strategy"
    return "policy_strategy"
