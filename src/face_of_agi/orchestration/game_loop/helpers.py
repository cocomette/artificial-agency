"""Small helpers shared by game-loop actions."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionSpec,
    AgentTrace,
    ContextDocuments,
    DecisionResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
)
from face_of_agi.models.adapters import OrchestratorAgentModel
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    AgentFrameworkInputCaptured,
    AgentProviderRequestsCaptured,
)
from face_of_agi.runtime import timing as runtime_timing


def decide_frame_turn(
    *,
    agent: OrchestratorAgentModel,
    contexts: ContextDocuments,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    tool_runtime: AgentToolRuntime | None,
    first_observation: Any | None,
    turn_id: int,
) -> tuple[DecisionResult, float]:
    """Return the frame decision, skipping Agent X for animation frames."""

    if not frame_context.control_mode.controllable:
        return synthetic_animation_decision(frame_context), 0.0

    if first_observation is None:
        raise RuntimeError("controllable frame is missing the first observation")

    debug.emit(
        AgentFrameworkInputCaptured(
            context=contexts.agent,
            world_game_context=contexts.world.game,
            goal_game_context=contexts.goal.game,
            first_observation=first_observation,
            current_observation=frame_context.current_observation,
            action_space=frame_context.control_mode.allowed_actions,
            recent_action_history=frame_context.recent_action_history,
            tool_runtime=tool_runtime,
        )
    )
    decision_started_at = perf_counter()
    with runtime_timing.span(
        "game_loop.agent_decide",
        step=frame_context.current_observation.step,
    ):
        decision = agent.decide(
            context=contexts.agent,
            first_observation=first_observation,
            current_observation=frame_context.current_observation,
            action_space=frame_context.control_mode.allowed_actions,
            tool_runtime=tool_runtime,
            world_game_context=contexts.world.game,
            goal_game_context=contexts.goal.game,
            recent_action_history=frame_context.recent_action_history,
        )
    decision_duration_seconds = perf_counter() - decision_started_at
    debug.capture_model_inputs(frame_context, turn_id, agent)
    debug.emit(
        AgentProviderRequestsCaptured(
            tuple(getattr(agent, "last_provider_requests", ()) or ())
        )
    )
    return decision, decision_duration_seconds


def synthetic_animation_decision(frame_context: FrameTurnContext) -> DecisionResult:
    """Build the orchestration-owned NONE decision for animation frames."""

    final_action = ActionSpec.none()
    trace = AgentTrace(
        step=frame_context.current_observation.step,
        first_observation_ref=frame_context.first_observation_ref,
        current_observation_ref=frame_context.current_observation_ref,
        final_action=final_action,
        reasoning_summary="non-controllable animation frame",
        metadata={
            "decision_source": "orchestration_synthetic_none",
            "agent_x_called": False,
        },
    )
    return DecisionResult(final_action=final_action, trace=trace)


def validate_decision(
    action: ActionSpec,
    *,
    control_mode: FrameControlMode,
) -> None:
    """Validate the chosen action against the current frame control mode."""

    if not control_mode.controllable:
        if not action.is_none():
            raise RuntimeError("non-final unrolled frame requires synthetic NONE action")
        return

    if action.is_none():
        raise RuntimeError("final controllable frame cannot submit synthetic NONE")

    is_allowed = any(
        candidate.action_id == action.action_id
        for candidate in control_mode.allowed_actions
    )
    if not is_allowed:
        raise RuntimeError(f"X returned invalid action for current frame: {action.name}")


def recent_action_history(
    action_history: list[ActionHistoryEntry],
    *,
    window: int,
) -> tuple[ActionHistoryEntry, ...]:
    """Return the bounded prior action history visible to X."""

    if window < 0:
        raise ValueError("action_history_window must be non-negative")
    if window == 0:
        return ()
    return tuple(action_history[-window:])


def build_action_history_entry(
    *,
    frame_context: FrameTurnContext,
    final_action: ActionSpec,
) -> ActionHistoryEntry:
    """Build one prompt-facing history entry after a valid frame decision."""

    return ActionHistoryEntry(
        action=final_action,
        controllable=frame_context.control_mode.controllable,
    )


def unroll_observation(observation: Observation) -> tuple[Observation, ...]:
    """Normalize one environment observation into ordered frame turns."""

    frames = observation.frames
    if not frames:
        frames = (observation.frame,)

    if len(frames) == 1:
        return (
            Observation(
                id=observation.id,
                step=observation.step,
                frame=frames[0],
                frames=(frames[0],),
                raw_frame_data=observation.raw_frame_data,
                metadata={
                    **observation.metadata,
                    "bundle_observation_id": observation.id,
                    "frame_index": 0,
                    "frame_count": 1,
                },
            ),
        )

    return tuple(
        Observation(
            id=f"{observation.id}-frame-{index}",
            step=observation.step,
            frame=frame,
            frames=(frame,),
            raw_frame_data=observation.raw_frame_data,
            metadata={
                **observation.metadata,
                "bundle_observation_id": observation.id,
                "frame_index": index,
                "frame_count": len(frames),
            },
        )
        for index, frame in enumerate(frames)
    )


def frame_control_mode(
    *,
    frame_index: int,
    frame_count: int,
    real_actions: tuple[ActionSpec, ...],
) -> FrameControlMode:
    """Return whether one unrolled frame can submit a real action."""

    if frame_index == frame_count - 1:
        return FrameControlMode.real_environment_turn(real_actions)
    return FrameControlMode.animation_unroll()
