"""Small helpers shared by game-loop actions."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

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
    history_anchor_observation: Any | None,
    turn_id: int,
) -> tuple[DecisionResult, float]:
    """Return the frame decision, skipping Agent X for animation frames."""

    if not frame_context.control_mode.controllable:
        return synthetic_animation_decision(frame_context), 0.0

    if history_anchor_observation is None:
        raise RuntimeError("controllable frame is missing the history anchor")

    debug.emit(
        AgentFrameworkInputCaptured(
            context=contexts.agent,
            history_anchor_observation=history_anchor_observation,
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
            history_anchor_observation=history_anchor_observation,
            current_observation=frame_context.current_observation,
            action_space=frame_context.control_mode.allowed_actions,
            tool_runtime=tool_runtime,
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


def agent_history_window(
    action_history: Sequence[ActionHistoryEntry],
    action_history_observations: Sequence[Observation],
    *,
    first_observation: Observation | None,
    window: int,
) -> tuple[Observation, tuple[ActionHistoryEntry, ...]]:
    """Return the visual anchor and matching bounded prompt history for X."""

    if first_observation is None:
        raise RuntimeError("frame turn is missing the first observation")
    if len(action_history) != len(action_history_observations):
        raise RuntimeError("action history observations are out of sync")
    if window < 0:
        raise ValueError("action_history_window must be non-negative")
    if window == 0 or not action_history:
        return first_observation, ()

    bounded_history = tuple(action_history[-window:])
    bounded_observations = tuple(action_history_observations[-window:])
    return bounded_observations[0], bounded_history


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
    input_frame_count = len(frames)
    frames = _drop_left_duplicate_frames(frames)

    if input_frame_count <= 1:
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


def _drop_left_duplicate_frames(frames: tuple[Any, ...]) -> tuple[Any, ...]:
    """Keep the rightmost frame from each consecutive identical run."""

    if len(frames) <= 1:
        return frames
    kept = [
        frame
        for frame, next_frame in zip(frames, frames[1:])
        if not _frames_equal(frame, next_frame)
    ]
    kept.append(frames[-1])
    return tuple(kept)


def _frames_equal(left: Any, right: Any) -> bool:
    """Return whether two raw game frames are exactly equal."""

    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if _numeric_array(left_array) and _numeric_array(right_array):
        if left_array.size == 0:
            return True
        difference = np.abs(
            left_array.astype("float64") - right_array.astype("float64")
        )
        return bool(np.max(difference) <= 0)
    return _structurally_equal(left, right)


def _numeric_array(array: Any) -> bool:
    """Return whether a numpy array can be diffed numerically."""

    import numpy as np

    return np.issubdtype(array.dtype, np.number)


def _structurally_equal(left: Any, right: Any) -> bool:
    """Return best-effort exact equality for non-numeric test fixtures."""

    try:
        equal = left == right
    except Exception:
        return False

    if isinstance(equal, bool):
        return equal

    try:
        import numpy as np

        return bool(np.all(equal))
    except Exception:
        return False


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
