"""Updater orchestration for frame turns and end-of-run contexts."""

from __future__ import annotations

from face_of_agi.contracts import (
    ContextDocuments,
    FrameTurnContext,
    ToolResult,
    UpdaterFrameTransitionInput,
)
from face_of_agi.memory import StateMemory
from face_of_agi.models.updater import (
    AgentGameContextUpdateInput,
    AgentProgressFeedback,
    GeneralKnowledgeUpdateInput,
    GoalGameContextUpdateInput,
    UpdaterTaskRegistry,
    WorldGameContextUpdateInput,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
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
    state_memory: StateMemory | None,
    frame_context: FrameTurnContext,
    turn_id: int,
) -> None:
    """Apply updater P to the live working contexts before persistence."""

    common_kwargs = {
        "current_observation_ref": update_input.current_observation_ref,
        "actual_next_observation_ref": update_input.actual_next_observation_ref,
        "post_decision_predictions": update_input.post_decision_predictions,
        "turn_metrics": update_input.turn_metrics,
        "submitted_action": update_input.submitted_action,
        "synthetic_none_action": update_input.synthetic_none_action,
        "metadata": dict(update_input.metadata),
    }

    world_update_input = WorldGameContextUpdateInput(
        previous_context=contexts.world,
        previous_observation=update_input.previous_observation,
        actual_next_observation=update_input.actual_next_observation,
        tool_results=tool_results_for_role(update_input, "world"),
        **common_kwargs,
    )
    debug.emit(UpdaterInputCaptured(role="world", update_input=world_update_input))
    world_updater = updater_tasks.require_world_game_updater()
    with runtime_timing.span("updater.world_game"):
        try:
            contexts.world = world_updater.update_world_game_context(
                world_update_input
            )
        finally:
            debug.emit(
                UpdaterProviderOutputCaptured(role="world", adapter=world_updater)
            )
    debug.capture_model_inputs(frame_context, turn_id, world_updater)

    goal_update_input = GoalGameContextUpdateInput(
        previous_context=contexts.goal,
        previous_observation=update_input.previous_observation,
        actual_next_observation=update_input.actual_next_observation,
        tool_results=tool_results_for_role(update_input, "goal"),
        **common_kwargs,
    )
    debug.emit(UpdaterInputCaptured(role="goal", update_input=goal_update_input))
    goal_updater = updater_tasks.require_goal_game_updater()
    with runtime_timing.span("updater.goal_game"):
        try:
            contexts.goal = goal_updater.update_goal_game_context(goal_update_input)
        finally:
            debug.emit(
                UpdaterProviderOutputCaptured(role="goal", adapter=goal_updater)
            )
    debug.capture_model_inputs(frame_context, turn_id, goal_updater)

    if update_input.actual_next_observation is None:
        raise ValueError("agent game updater requires the current observation")
    previous_world_context = (
        state_memory.read_previous_world_game_context(
            game_id=frame_context.game_id,
            before_state_id=frame_context.previous_source_state_id,
        )
        if state_memory is not None
        else None
    )
    agent_update_input = AgentGameContextUpdateInput(
        previous_context=contexts.agent,
        previous_observation=update_input.previous_observation,
        current_observation=update_input.actual_next_observation,
        current_turn_world_game_context=world_update_input.previous_context.game,
        current_turn_goal_game_context=goal_update_input.previous_context.game,
        previous_turn_world_game_context=previous_world_context,
        trace=update_input.decision_trace,
        turn_metrics=AgentProgressFeedback(
            time_cost=update_input.turn_metrics.time_cost,
            score_delta=update_input.turn_metrics.score_delta,
        ),
    )
    debug.emit(UpdaterInputCaptured(role="agent", update_input=agent_update_input))
    agent_updater = updater_tasks.require_agent_game_updater()
    with runtime_timing.span("updater.agent_game"):
        try:
            contexts.agent = agent_updater.update_agent_game_context(agent_update_input)
        finally:
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
    world_update_input = GeneralKnowledgeUpdateInput(
        role="world",
        previous_context=contexts.world,
        **common_kwargs,
    )
    debug.emit(UpdaterInputCaptured(role="world", update_input=world_update_input))
    with runtime_timing.span("updater.general_world"):
        try:
            contexts.world = general_updater.update_general_knowledge(
                world_update_input
            )
        finally:
            debug.emit(
                UpdaterProviderOutputCaptured(role="world", adapter=general_updater)
            )

    goal_update_input = GeneralKnowledgeUpdateInput(
        role="goal",
        previous_context=contexts.goal,
        **common_kwargs,
    )
    debug.emit(UpdaterInputCaptured(role="goal", update_input=goal_update_input))
    with runtime_timing.span("updater.general_goal"):
        try:
            contexts.goal = general_updater.update_general_knowledge(goal_update_input)
        finally:
            debug.emit(
                UpdaterProviderOutputCaptured(role="goal", adapter=general_updater)
            )

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


def tool_results_for_role(
    update_input: UpdaterFrameTransitionInput,
    role: str,
) -> tuple[ToolResult, ...]:
    """Return live trace tool results for one updater role."""

    return tuple(
        result
        for result in update_input.decision_trace.tool_results
        if result.tool == role
    )
