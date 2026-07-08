"""Small helpers shared by game-loop actions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionOutcomeEvidence,
    ActionSpec,
    AgentTrace,
    ChangeSummaryElement,
    ContextDocuments,
    DecisionResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
)
from face_of_agi.models.action_history import model_facing_action_text
from face_of_agi.models.adapters import OrchestratorAgentModel
from face_of_agi.models.memory import GameMemoryDocument
from face_of_agi.models.orchestrator_agent import AgentToolRuntime
from face_of_agi.debug.bus import DebugBus
from face_of_agi.debug.events import (
    AgentFrameworkInputCaptured,
    AgentProviderRequestsCaptured,
    ModelCallCompleted,
)
from face_of_agi.orchestration.game_loop.fallbacks import (
    fallback_decision_result,
    is_agent_model_failure,
)
from face_of_agi.models.providers.scheduler import model_call_context
from face_of_agi.runtime import timing as runtime_timing

_NO_ANCHOR = object()
_AGENT_REPAIR_ATTEMPTS_PATTERN = re.compile(r"after (\d+) repair attempt")


@dataclass(frozen=True, slots=True)
class PromptActionOutcome:
    """Prompt-facing allowed-actions filter plus updater evidence."""

    allowed_actions: tuple[ActionSpec, ...]
    evidence: ActionOutcomeEvidence


@dataclass(frozen=True, slots=True)
class _FrameCandidate:
    """One raw bundle frame after consecutive duplicate filtering."""

    frame: Any
    original_index: int


@dataclass(frozen=True, slots=True)
class _SelectedFrame:
    """One retained frame plus skipped raw-frame count since the prior source."""

    frame: Any
    original_index: int
    skipped_intermediate_frame_count: int


def decide_frame_turn(
    *,
    agent: OrchestratorAgentModel,
    contexts: ContextDocuments,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    recent_action_history_available: bool,
    tool_runtime: AgentToolRuntime | None,
    turn_id: int,
    action_suppression_zero_changed_pixel_turns: int,
    game_memory: GameMemoryDocument,
) -> tuple[DecisionResult, float]:
    """Return the frame decision, skipping Agent X for animation frames."""

    if not frame_context.control_mode.controllable:
        return synthetic_animation_decision(frame_context), 0.0

    prompt_actions = prompt_action_outcome(
        action_space=frame_context.control_mode.allowed_actions,
        action_history=frame_context.recent_action_history,
        action_suppression_zero_changed_pixel_turns=(
            action_suppression_zero_changed_pixel_turns
        ),
        updater_stagnation_warning_zero_changed_pixel_turns=0,
        crop_box_normalized=model_input_crop_box_normalized(agent),
    )
    debug.emit(
        AgentFrameworkInputCaptured(
            context=contexts.agent,
            current_observation=frame_context.current_observation,
            action_space=prompt_actions.allowed_actions,
            recent_action_history=frame_context.recent_action_history,
            tool_runtime=tool_runtime,
        )
    )
    decision_started_at = perf_counter()
    decision: DecisionResult | None = None
    repair_attempts = 0
    with runtime_timing.span(
        "game_loop.agent_decide",
        step=frame_context.current_observation.step,
    ):
        try:
            try:
                with model_call_context(
                    run_id=frame_context.run_id,
                    game_id=frame_context.game_id,
                    turn_id=turn_id,
                    role="agent",
                    emit_event=debug.record_model_call_event,
                ):
                    decision = agent.decide(
                        context=contexts.agent,
                        current_observation=frame_context.current_observation,
                        action_space=prompt_actions.allowed_actions,
                        tool_runtime=tool_runtime,
                        recent_action_history=frame_context.recent_action_history,
                        glossary_actions=frame_context.control_mode.allowed_actions,
                        first_observation_ref=frame_context.first_observation_ref,
                        recent_action_history_available=recent_action_history_available,
                        action_outcome_evidence=prompt_actions.evidence,
                        game_memory=game_memory,
                    )
            except Exception as exc:
                if not is_agent_model_failure(exc):
                    raise
                repair_attempts = agent_repair_attempts_from_error(exc)
                decision = fallback_decision_result(
                    frame_context=frame_context,
                    turn_id=turn_id,
                    action_space=prompt_actions.allowed_actions,
                    error=exc,
                )
                if repair_attempts:
                    decision.trace.metadata["repair_count"] = repair_attempts
        finally:
            repair_attempts = repair_attempts_from_metadata(
                decision.trace if decision is not None else None,
                default=repair_attempts,
            )
            decision_duration_seconds = perf_counter() - decision_started_at
            debug.emit(
                ModelCallCompleted(
                    role="agent",
                    duration_seconds=decision_duration_seconds,
                    repair_attempts=repair_attempts,
                    game_id=frame_context.game_id,
                    turn_id=turn_id,
                )
            )
    debug.capture_model_inputs(frame_context, turn_id, agent)
    debug.emit(
        AgentProviderRequestsCaptured(
            tuple(getattr(agent, "last_provider_requests", ()) or ())
        )
    )
    return decision, decision_duration_seconds


def repair_attempts_from_metadata(
    value: Any,
    *,
    default: int = 0,
) -> int:
    """Return a non-negative repair count from result or trace metadata."""

    metadata = getattr(value, "metadata", value)
    fallback = _non_negative_int(default)
    if not isinstance(metadata, dict):
        return fallback
    for key in ("repair_attempts", "repair_count"):
        if key not in metadata:
            continue
        try:
            return max(0, int(metadata[key]))
        except (TypeError, ValueError):
            return fallback
    return fallback


def agent_repair_attempts_from_error(error: Exception) -> int:
    """Extract Agent X repair exhaustion count from adapter error text."""

    match = _AGENT_REPAIR_ATTEMPTS_PATTERN.search(str(error))
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return 0


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


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
    if action.name == "ACTION6":
        _action6_arc_grid_coordinate(action, "x")
        _action6_arc_grid_coordinate(action, "y")
        if action.target is None or not action.target.strip():
            raise RuntimeError("ACTION6 requires a non-empty target")


def bounded_agent_action_history(
    action_history: Sequence[ActionHistoryItem],
    *,
    window: int,
) -> tuple[ActionHistoryItem, ...]:
    """Return bounded prompt-facing action history for X."""

    return bounded_action_history(
        action_history,
        window=window,
        key="agent_action_history_window",
    )


def bounded_action_history(
    action_history: Sequence[ActionHistoryItem],
    *,
    window: int,
    key: str,
) -> tuple[ActionHistoryItem, ...]:
    """Return latest controllable action groups allowed by one config window."""

    if window < 0:
        raise ValueError(f"{key} must be non-negative")
    if window == 0 or not action_history:
        return ()
    groups_seen = 0
    for index in range(len(action_history) - 1, -1, -1):
        item = action_history[index]
        if isinstance(item, ActionHistoryEntry) and item.controllable:
            groups_seen += 1
            if groups_seen == window:
                return tuple(action_history[index:])
    return tuple(action_history)


def prompt_action_outcome(
    *,
    action_space: Sequence[ActionSpec],
    action_history: Sequence[ActionHistoryItem],
    action_suppression_zero_changed_pixel_turns: int,
    updater_stagnation_warning_zero_changed_pixel_turns: int,
    crop_box_normalized: Any | None = None,
) -> PromptActionOutcome:
    """Return prompt-facing allowed actions plus low-information evidence."""

    actions = tuple(action_space)
    if action_suppression_zero_changed_pixel_turns < 0:
        raise ValueError(
            "action_suppression_zero_changed_pixel_turns must be non-negative"
        )
    if updater_stagnation_warning_zero_changed_pixel_turns < 0:
        raise ValueError(
            "updater_stagnation_warning_zero_changed_pixel_turns must be non-negative"
        )

    controllable_history = _controllable_entries_since_last_reset(action_history)
    suppressed_actions: tuple[str, ...] = ()
    suppression_reason = ""
    disabled_reason = ""
    repeated_action = ""
    repeated_count = 0
    suppress_action6_as_class = _suppress_action6_as_class(actions)

    latest_streak = _latest_same_action_streak(
        controllable_history,
        suppress_action6_as_class=suppress_action6_as_class,
    )
    latest_same_action_zero_changed_pixel_count = (
        _latest_same_action_zero_changed_pixel_count(latest_streak)
    )
    if latest_streak:
        repeated_action = _action_suppression_label(
            latest_streak[0].action,
            suppress_action6_as_class=suppress_action6_as_class,
            crop_box_normalized=crop_box_normalized,
        )
        repeated_count = len(latest_streak)

    if (
        action_suppression_zero_changed_pixel_turns > 0
        and len(latest_streak) >= action_suppression_zero_changed_pixel_turns
    ):
        latest_action = latest_streak[0].action
        latest_window = latest_streak[:action_suppression_zero_changed_pixel_turns]
        allowed_match = any(
            candidate.name == latest_action.name for candidate in actions
        )
        if (
            allowed_match
            and _is_suppressible_prompt_action(latest_action)
            and all(entry.changed_pixel_count == 0 for entry in latest_window)
        ):
            suppression_label = _action_suppression_label(
                latest_action,
                suppress_action6_as_class=suppress_action6_as_class,
                crop_box_normalized=crop_box_normalized,
            )
            if latest_action.name == "ACTION6" and not suppress_action6_as_class:
                suppressed_actions = (suppression_label,)
                suppression_reason = (
                    f"{suppression_label} was prompt-suppressed because the "
                    f"latest {action_suppression_zero_changed_pixel_turns} "
                    "controllable uses of that coordinate had "
                    "changed_pixels=0. ACTION6 remains available; choose a "
                    "different coordinate."
                )
            else:
                filtered = tuple(
                    action
                    for action in actions
                    if action.name != latest_action.name
                )
                if filtered:
                    actions = filtered
                    suppressed_actions = (suppression_label,)
                    suppression_reason = (
                        f"{suppression_label} was omitted because the latest "
                        f"{action_suppression_zero_changed_pixel_turns} controllable "
                        "uses of that action had changed_pixels=0."
                    )
                else:
                    disabled_reason = (
                        "suppression skipped because it would remove every currently "
                        "allowed action"
                    )

    stagnation_warning = (
        updater_stagnation_warning_zero_changed_pixel_turns > 0
        and latest_same_action_zero_changed_pixel_count
        >= updater_stagnation_warning_zero_changed_pixel_turns
    )
    return PromptActionOutcome(
        allowed_actions=actions,
        evidence=ActionOutcomeEvidence(
            suppression_threshold=action_suppression_zero_changed_pixel_turns,
            suppressed_actions=suppressed_actions,
            suppression_reason=suppression_reason,
            suppression_disabled_reason=disabled_reason,
            latest_repeated_action=repeated_action,
            latest_repeated_action_count=repeated_count,
            latest_same_action_zero_changed_pixel_turn_count=(
                latest_same_action_zero_changed_pixel_count
            ),
            stagnation_warning_threshold=(
                updater_stagnation_warning_zero_changed_pixel_turns
            ),
            stagnation_warning=stagnation_warning,
        ),
    )


def _latest_same_action_streak(
    history: Sequence[ActionHistoryEntry],
    *,
    suppress_action6_as_class: bool,
) -> tuple[ActionHistoryEntry, ...]:
    if not history:
        return ()
    latest_identity = _action_suppression_identity(
        history[-1].action,
        suppress_action6_as_class=suppress_action6_as_class,
    )
    streak: list[ActionHistoryEntry] = []
    for entry in reversed(history):
        if (
            _action_suppression_identity(
                entry.action,
                suppress_action6_as_class=suppress_action6_as_class,
            )
            != latest_identity
        ):
            break
        streak.append(entry)
    return tuple(streak)


def _controllable_entries_since_last_reset(
    history: Sequence[ActionHistoryItem],
) -> tuple[ActionHistoryEntry, ...]:
    latest_run_items: Sequence[ActionHistoryItem] = history
    for index in range(len(history) - 1, -1, -1):
        if isinstance(history[index], ActionHistoryResetMarker):
            latest_run_items = history[index + 1 :]
            break
    return tuple(
        entry
        for entry in latest_run_items
        if isinstance(entry, ActionHistoryEntry) and entry.controllable
    )


def _latest_same_action_zero_changed_pixel_count(
    history: Sequence[ActionHistoryEntry],
) -> int:
    count = 0
    for entry in history:
        if entry.changed_pixel_count != 0:
            break
        count += 1
    return count


def _is_suppressible_prompt_action(action: ActionSpec) -> bool:
    return (
        not action.is_none()
        and (
            action.name == "ACTION6"
            or (action.data is None and not action.is_complex())
        )
    )


def _suppress_action6_as_class(actions: Sequence[ActionSpec]) -> bool:
    has_action6 = any(action.name == "ACTION6" for action in actions)
    has_non_action6 = any(action.name != "ACTION6" for action in actions)
    return has_action6 and has_non_action6


def _action_suppression_identity(
    action: ActionSpec,
    *,
    suppress_action6_as_class: bool,
) -> tuple[Any, ...]:
    if action.name == "ACTION6":
        if suppress_action6_as_class:
            return ("ACTION6",)
        return (
            "ACTION6",
            _action6_arc_grid_coordinate(action, "x"),
            _action6_arc_grid_coordinate(action, "y"),
        )
    if action.data:
        return (action.name, tuple(sorted(action.data.items())))
    return (action.name,)


def _action_suppression_label(
    action: ActionSpec,
    *,
    suppress_action6_as_class: bool,
    crop_box_normalized: Any | None = None,
) -> str:
    if action.name == "ACTION6" and not suppress_action6_as_class:
        return model_facing_action_text(
            action,
            crop_box_normalized=crop_box_normalized,
        )
    return action.name


def model_input_crop_box_normalized(model: Any) -> Any | None:
    """Return the prompt image crop box exposed by a model adapter, if any."""

    crop_box = getattr(model, "input_image_crop_box_normalized", None)
    if crop_box is not None:
        return crop_box
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "input_image_crop_box_normalized", None)


def _action6_arc_grid_coordinate(action: ActionSpec, key: str) -> int:
    if action.data is None or key not in action.data:
        raise ValueError(f"ACTION6 data missing {key!r}")
    value = action.data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ACTION6 data {key!r} must be numeric")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"ACTION6 data {key!r} must be an ARC grid integer")
    if not 0 <= numeric <= 63:
        raise ValueError(f"ACTION6 data {key!r} must be in ARC grid 0..63")
    return int(numeric)


def build_action_history_entry(
    *,
    frame_context: FrameTurnContext,
    final_action: ActionSpec,
    next_observation: Observation,
    changed_pixel_count: int,
    change_summary: str,
    change_elements: tuple[ChangeSummaryElement, ...] = (),
    changed_pixel_percent: float | None = None,
    completed_levels: int | None = None,
    action_count: int | None = None,
) -> ActionHistoryEntry:
    """Build one prompt-facing history entry after a valid frame decision."""

    if changed_pixel_count < 0:
        raise ValueError("changed_pixel_count must be non-negative")
    return ActionHistoryEntry(
        action=final_action,
        controllable=frame_context.control_mode.controllable,
        changed_pixel_count=changed_pixel_count,
        change_summary=change_summary,
        change_elements=change_elements,
        changed_pixel_percent=changed_pixel_percent,
        completed_levels=completed_levels,
        action_count=action_count,
        skipped_intermediate_animation_frame_count=(
            skipped_intermediate_animation_frame_count(
                frame_context,
                next_observation=next_observation,
            )
        ),
    )


def changed_pixel_count(left: Any, right: Any) -> int:
    """Return the raw frame cell/pixel count changed between two frames."""

    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return max(_frame_surface_size(left_array), _frame_surface_size(right_array))
    if left_array.shape == ():
        return 0 if _structurally_equal(left, right) else 1
    if _numeric_array(left_array) and _numeric_array(right_array):
        changed = left_array != right_array
        if _rgb_like_array(left_array):
            changed = np.any(changed, axis=-1)
        return int(np.count_nonzero(changed))
    changed = left_array != right_array
    if _rgb_like_array(left_array):
        changed = np.any(changed, axis=-1)
    return int(np.count_nonzero(changed))


def unroll_observation(
    observation: Observation,
    *,
    animation_keyframe_pixel_threshold: int = 8,
    anchor_frame: Any = _NO_ANCHOR,
) -> tuple[Observation, ...]:
    """Normalize one environment observation into ordered frame turns."""

    if animation_keyframe_pixel_threshold < 0:
        raise ValueError("animation_keyframe_pixel_threshold must be non-negative")

    frames = observation.frames
    if not frames:
        frames = (observation.frame,)
    input_frame_count = len(frames)
    selected_frames = _select_animation_keyframes(
        frames,
        threshold=animation_keyframe_pixel_threshold,
        anchor_frame=anchor_frame,
    )

    if input_frame_count <= 1:
        selected = selected_frames[0]
        return (
            Observation(
                id=observation.id,
                step=observation.step,
                frame=selected.frame,
                frames=(selected.frame,),
                raw_frame_data=observation.raw_frame_data,
                metadata={
                    **observation.metadata,
                    "bundle_observation_id": observation.id,
                    "frame_index": 0,
                    "frame_count": 1,
                    "bundle_frame_index": selected.original_index,
                    "skipped_intermediate_animation_frame_count": (
                        selected.skipped_intermediate_frame_count
                    ),
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
                "frame_count": len(selected_frames),
                "bundle_frame_index": selected.original_index,
                "skipped_intermediate_animation_frame_count": (
                    selected.skipped_intermediate_frame_count
                ),
            },
        )
        for index, selected in enumerate(selected_frames)
        for frame in (selected.frame,)
    )


def skipped_intermediate_animation_frame_count(
    frame_context: FrameTurnContext,
    *,
    next_observation: Observation | None = None,
) -> int:
    """Return the count of collapsed intermediate animation frames for history."""

    key = "skipped_intermediate_animation_frame_count"
    if next_observation is not None and key in next_observation.metadata:
        value = next_observation.metadata.get(
            key,
            0,
        )
    else:
        value = frame_context.current_observation.metadata.get(
            key,
            0,
        )
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _select_animation_keyframes(
    frames: tuple[Any, ...],
    *,
    threshold: int,
    anchor_frame: Any,
) -> tuple[_SelectedFrame, ...]:
    """Return retained bundle keyframes after duplicate and threshold filtering."""

    candidates = _drop_left_duplicate_frames(frames)
    if not candidates:
        return ()

    selected: list[_SelectedFrame] = []
    if anchor_frame is _NO_ANCHOR:
        first = candidates[0]
        selected.append(
            _SelectedFrame(
                frame=first.frame,
                original_index=first.original_index,
                skipped_intermediate_frame_count=0,
            )
        )
        anchor = first.frame
        last_retained_index = first.original_index
        remaining = candidates[1:]
    else:
        anchor = anchor_frame
        last_retained_index = -1
        remaining = candidates

    for candidate in remaining:
        if threshold == 0 or changed_pixel_count(anchor, candidate.frame) >= threshold:
            selected_frame = _selected_frame_since(
                candidate,
                last_retained_index=last_retained_index,
            )
            selected.append(selected_frame)
            anchor = candidate.frame
            last_retained_index = candidate.original_index

    final_candidate = candidates[-1]
    if (
        not selected
        or selected[-1].original_index != final_candidate.original_index
    ):
        selected.append(
            _selected_frame_since(
                final_candidate,
                last_retained_index=last_retained_index,
            )
        )

    return tuple(selected)


def _selected_frame_since(
    candidate: _FrameCandidate,
    *,
    last_retained_index: int,
) -> _SelectedFrame:
    return _SelectedFrame(
        frame=candidate.frame,
        original_index=candidate.original_index,
        skipped_intermediate_frame_count=max(
            0,
            candidate.original_index - last_retained_index - 1,
        ),
    )


def _drop_left_duplicate_frames(frames: tuple[Any, ...]) -> tuple[_FrameCandidate, ...]:
    """Keep the rightmost frame from each consecutive identical run."""

    if len(frames) <= 1:
        return tuple(
            _FrameCandidate(frame=frame, original_index=index)
            for index, frame in enumerate(frames)
        )
    kept = [
        _FrameCandidate(frame=frame, original_index=index)
        for index, (frame, next_frame) in enumerate(zip(frames, frames[1:]))
        if not _frames_equal(frame, next_frame)
    ]
    kept.append(_FrameCandidate(frame=frames[-1], original_index=len(frames) - 1))
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


def _rgb_like_array(array: Any) -> bool:
    """Return whether a raw frame array stores color channels per pixel."""

    return array.ndim == 3 and array.shape[-1] in {3, 4}


def _frame_surface_size(array: Any) -> int:
    """Return countable raw cells/pixels for one frame array."""

    if array.shape == ():
        return 1
    if _rgb_like_array(array):
        return int(array.shape[0] * array.shape[1])
    return int(array.size)


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
    return FrameControlMode.animation_unroll(real_actions)
