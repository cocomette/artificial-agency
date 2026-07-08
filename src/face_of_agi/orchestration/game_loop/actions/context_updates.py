"""Updater orchestration for frame turns and end-of-run contexts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from time import perf_counter

from face_of_agi.contracts import (
    ActionHistoryItem,
    ActionSpec,
    ContextDocuments,
    FrameTurnContext,
    Observation,
    RoleContext,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models.updater import (
    AGENT_GAME_CONTEXT_KEYS,
    AgentGameContextUpdateInput,
    AgentGameContextUpdateResult,
    UpdaterTaskRegistry,
)
from face_of_agi.models.compacter import (
    AgentCompacterSummary,
    AgentCompacterModel,
    AgentCompacterInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    ModelCallCompleted,
    UpdaterInputCaptured,
    UpdaterProviderOutputCaptured,
)
from face_of_agi.runtime import timing as runtime_timing

def apply_context_updates(
    update_input: UpdaterFrameTransitionInput,
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    compacter_context: AgentCompacterSummary,
    game_last_started_turns_ago: int | None,
    game_start_reason: str | None,
    updater_actions_window: int,
    turn_id: int,
    action_history: Sequence[ActionHistoryItem],
    strategy_history: Sequence[str] = (),
    previous_level_summary: str = "",
    reset_notice: str = "",
) -> tuple[ActionSpec, ...]:
    """Apply updater P to the live working contexts before persistence."""

    if update_input.actual_next_observation is None:
        raise ValueError("game updaters require the current observation")
    if update_input.action_history_entry is None:
        raise ValueError("game updaters require a current action history entry")

    result = apply_agent_context_update(
        contexts=contexts,
        updater_tasks=updater_tasks,
        debug=debug,
        frame_context=frame_context,
        current_observation=update_input.actual_next_observation,
        action_history=action_history,
        allowed_action_source=frame_context.control_mode.allowed_actions,
        compacter_context=compacter_context,
        previous_game_context_history=tuple(strategy_history),
        previous_level_summary=previous_level_summary,
        turn_id=turn_id,
        updater_actions_window=updater_actions_window,
        reset_notice=reset_notice,
    )
    return result.next_actions


def apply_agent_context_update(
    *,
    contexts: ContextDocuments,
    updater_tasks: UpdaterTaskRegistry,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    current_observation: Observation,
    action_history: Sequence[ActionHistoryItem],
    allowed_action_source: Sequence[ActionSpec],
    compacter_context: AgentCompacterSummary | None = None,
    previous_game_context_history: tuple[str, ...] = (),
    turn_id: int,
    updater_actions_window: int = 1,
    previous_level_summary: str = "",
    reset_notice: str = "",
) -> AgentGameContextUpdateResult:
    """Run the agent updater and merge its strategy fields."""

    agent_update_input = AgentGameContextUpdateInput(
        current_observation=current_observation,
        allowed_actions=tuple(allowed_action_source),
        glossary_actions=tuple(allowed_action_source),
        world_model_context=_compacter_context_text(compacter_context),
        previous_actions_summary=_previous_actions_summary_text(compacter_context),
        previous_strategy_summary=(
            previous_level_summary
            or _previous_strategy_summary_text(compacter_context)
        ),
        action_history=tuple(action_history),
        previous_game_context_history=tuple(previous_game_context_history),
        reset_notice=reset_notice,
        actions_window=updater_actions_window,
    )
    task_label = "agent"
    debug.emit(UpdaterInputCaptured(role=task_label, update_input=agent_update_input))
    agent_updater = updater_tasks.require_agent_updater()
    span_name = "updater.agent"
    update = agent_updater.update_agent_context
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
    if len(result.next_actions) != agent_update_input.actions_window:
        raise RuntimeError(
            f"{task_label} updater returned {len(result.next_actions)} actions, "
            f"expected exactly {agent_update_input.actions_window}"
        )
    updated_fields = _agent_context_fields(result.context)
    contexts.agent = RoleContext(
        general=contexts.agent.general,
        game=_dump_agent_context_fields(updated_fields),
    )
    return result


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


def previous_level_summary_text(
    *,
    state_memory: StateMemory | None,
    run_id: str,
    game_id: str,
) -> str:
    """Return the latest solved-level strategy summary as updater text."""

    summary = _latest_compacter_level_summary(
        state_memory,
        run_id=run_id,
        game_id=game_id,
    )
    if summary is None:
        return ""
    return str(getattr(summary, "previous_strategy_summary"))


def latest_compacter_summary_after_state_id(
    *,
    state_memory: StateMemory | None,
    run_id: str,
    game_id: str,
) -> int | None:
    """Return the state id after which the current compacter level starts."""

    return _latest_compacter_summary_after_state_id(
        _latest_compacter_level_summary(
            state_memory,
            run_id=run_id,
            game_id=game_id,
        )
    )


def compact_agent_context(
    *,
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    compacter: AgentCompacterModel | None,
    current_observation: Observation,
    action_history: Sequence[ActionHistoryItem],
    strategy_history: Sequence[str],
    allowed_actions: Sequence[ActionSpec],
    turn_id: int,
    debug: DebugBus,
) -> AgentCompacterSummary:
    """Run the compacter for the current frame turn."""

    if compacter is None:
        raise RuntimeError("compacter model is not registered")
    previous_compacter_context = (
        ""
        if state_memory is None
        else state_memory.read_compacter_context_before(
            game_id=frame_context.game_id,
            before_state_id=frame_context.current_source_state_id,
        )
    )
    compacter_input = AgentCompacterInput(
        game_id=frame_context.game_id,
        previous_compacter_context=previous_compacter_context,
        current_observation=current_observation,
        action_history=tuple(action_history),
        strategy_history=tuple(strategy_history),
        allowed_actions=tuple(allowed_actions),
        metadata={
            "run_id": frame_context.run_id,
            "before_state_id": frame_context.current_source_state_id,
        },
    )
    with runtime_timing.span(
        "compacter.agent_context",
        turn_id=turn_id,
    ):
        started_at = perf_counter()
        try:
            return compacter.compact_agent_context(compacter_input)
        finally:
            debug.emit(
                ModelCallCompleted(
                    role="compacter",
                    duration_seconds=perf_counter() - started_at,
                )
            )


def agent_context_strategy_snapshot(contexts: ContextDocuments) -> dict[str, str]:
    """Return the current agent strategy context for history storage."""

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
        raise RuntimeError("agent game context must be JSON") from None
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


def _dump_agent_context_fields(fields: dict[str, Any]) -> str:
    return json.dumps(
        {key: fields[key] for key in fields},
        indent=2,
        ensure_ascii=False,
    )


def _compacter_context_text(summary: AgentCompacterSummary | str | None) -> str:
    if summary is None:
        return "not available"
    if isinstance(summary, str):
        return summary or "not available"
    action_lines = [
        f"- {key}: {_text_or_none(value)}"
        for key, value in summary.action_effects.items()
    ]
    return "\n".join(
        [
            "world_description: " + _text_or_none(summary.world_description),
            "special_events: " + _text_or_none(summary.special_events),
            "action_effects:\n" + (
                "\n".join(action_lines) if action_lines else "not available"
            ),
        ]
    )


def _previous_actions_summary_text(summary: AgentCompacterSummary | None) -> str:
    if summary is None:
        return "not available"
    return _text_or_none(summary.previous_actions_summary)


def _previous_strategy_summary_text(summary: AgentCompacterSummary | None) -> str:
    if summary is None:
        return "not available"
    return _text_or_none(summary.previous_strategy_summary)


def _latest_compacter_level_summary(
    state_memory: StateMemory | None,
    *,
    run_id: str,
    game_id: str,
) -> object | None:
    if state_memory is None:
        return None
    return state_memory.read_latest_compacter_level_summary(
        run_id=run_id,
        game_id=game_id,
    )


def _latest_compacter_summary_after_state_id(summary: object | None) -> int | None:
    if summary is None:
        return None
    source_state_ids = getattr(summary, "source_state_ids")
    if not source_state_ids:
        return None
    return int(source_state_ids[-1])


def _text_or_none(value: str | None) -> str:
    if value is None:
        return "none"
    text = value.strip()
    return text if text else "none"
