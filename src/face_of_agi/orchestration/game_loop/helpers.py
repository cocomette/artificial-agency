"""Small helpers shared by game-loop actions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Sequence

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionSpec,
    AgentTrace,
    DecisionResult,
    FrameControlMode,
    FrameTurnContext,
    Observation,
)
from face_of_agi.debug.bus import DebugBus
from face_of_agi.models.arc_grid_crop import (
    ARC_GRID_SIZE,
    crop_image_arc_grid_edges,
    normalize_arc_grid_crop_edges,
)
from face_of_agi.frames import frame_to_pil_image, observation_to_pil_image
from face_of_agi.runtime import timing as runtime_timing

UNCHANGED_FRAME_CHANGE_SUMMARY = (
    "No changes happened for this transition. "
    "The previous and current frames are identical"
)


@dataclass(frozen=True, slots=True)
class _FrameCandidate:
    """One raw bundle frame after consecutive duplicate filtering."""

    frame: Any
    original_index: int


@dataclass(frozen=True, slots=True)
class _ObservationCandidate:
    """One frozen visual observation candidate for transition evidence."""

    observation: Observation
    original_index: int


@dataclass(frozen=True, slots=True)
class _SelectedFrame:
    """One retained frame plus skipped raw-frame count since the prior source."""

    frame: Any
    original_index: int
    skipped_intermediate_frame_count: int


def decide_frame_turn(
    *,
    debug: DebugBus,
    frame_context: FrameTurnContext,
    queued_actions: tuple[ActionSpec, ...],
    queued_updater_mode: str | None,
) -> tuple[DecisionResult, float]:
    """Return the controllable frame decision from the stored updater action."""

    if not frame_context.control_mode.controllable:
        return synthetic_animation_decision(frame_context), 0.0

    if not queued_actions:
        raise RuntimeError("controllable frame is missing updater action")
    queued_action = queued_actions[0]
    decision_started_at = perf_counter()
    with runtime_timing.span("game_loop.updater_decision"):
        decision = updater_action_decision(
            frame_context=frame_context,
            queued_action=queued_action,
            updater_mode=queued_updater_mode,
        )
    decision_duration_seconds = perf_counter() - decision_started_at
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


def updater_action_decision(
    *,
    frame_context: FrameTurnContext,
    queued_action: ActionSpec,
    updater_mode: str | None,
) -> DecisionResult:
    """Wrap an updater-selected action in the normal decision trace shape."""

    trace = AgentTrace(
        step=frame_context.current_observation.step,
        first_observation_ref=frame_context.first_observation_ref,
        current_observation_ref=frame_context.current_observation_ref,
        final_action=queued_action,
        reasoning_summary="updater-selected action",
        metadata={
            "decision_source": "agent_updater",
            "updater_mode": updater_mode,
            "agent_x_called": False,
        },
    )
    return DecisionResult(final_action=queued_action, trace=trace)


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

    if not action_allowed(action, control_mode=control_mode):
        raise RuntimeError(
            f"updater returned invalid action for current frame: {action.name}"
        )


def action_allowed(
    action: ActionSpec,
    *,
    control_mode: FrameControlMode,
) -> bool:
    """Return whether an action id is available on the current controllable frame."""

    return any(
        candidate.action_id == action.action_id
        for candidate in control_mode.allowed_actions
    )


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


def build_action_history_entry(
    *,
    frame_context: FrameTurnContext,
    final_action: ActionSpec,
    next_observation: Observation,
    previous_observation: Observation | None = None,
    change_summary: str,
    change_elements: tuple[Any, ...] = (),
    change_summary_crop_edges: Any | None = None,
    supplied_changed_pixel_count: float | None = None,
    completed_levels: int | None = None,
    action_count: int | None = None,
    action_mode: str | None = None,
    controllable: bool | None = None,
    animation_frame_count: int | None = None,
    avg_changed_pixel_count: float | None = None,
) -> ActionHistoryEntry:
    """Build one prompt-facing history entry after a valid frame decision."""

    return ActionHistoryEntry(
        action=final_action,
        controllable=(
            frame_context.control_mode.controllable
            if controllable is None
            else controllable
        ),
        changed_pixel_count=(
            supplied_changed_pixel_count
            if supplied_changed_pixel_count is not None
            else change_summary_visible_changed_pixel_count(
                previous_observation or frame_context.current_observation,
                next_observation,
                crop_edges=change_summary_crop_edges,
            )
        ),
        change_summary=change_summary,
        change_elements=change_elements,
        completed_levels=completed_levels,
        action_count=action_count,
        action_mode=_action_mode(action_mode),
        skipped_intermediate_animation_frame_count=(
            skipped_intermediate_animation_frame_count(
                frame_context,
                next_observation=next_observation,
            )
        ),
        animation_frame_count=animation_frame_count,
        avg_changed_pixel_count=avg_changed_pixel_count,
    )


def change_summary_visible_changed_pixel_count(
    previous_observation: Observation,
    current_observation: Observation,
    *,
    crop_edges: Any | None,
) -> float:
    """Return visible frame changes as a percentage of cropped frame area."""

    edges = normalize_arc_grid_crop_edges(crop_edges)
    previous_frame = _change_count_frame(previous_observation, edges)
    current_frame = _change_count_frame(current_observation, edges)
    visible_surface = max(
        _frame_surface_size_for_count(previous_frame),
        _frame_surface_size_for_count(current_frame),
    )
    if visible_surface <= 0:
        return 0.0
    changed = changed_pixel_count(previous_frame, current_frame)
    if changed <= 0:
        return 0.0
    percentage = changed * 100.0 / visible_surface
    return max(round(percentage, 4), 0.0001)


def average_consecutive_visible_changed_pixel_count(
    observations: Sequence[Observation],
    *,
    crop_edges: Any | None,
) -> float:
    """Return average consecutive visible changed-pixel percentage."""

    if len(observations) < 2:
        return 0.0
    changes = [
        change_summary_visible_changed_pixel_count(
            previous,
            current,
            crop_edges=crop_edges,
        )
        for previous, current in zip(observations, observations[1:])
    ]
    if not changes:
        return 0.0
    return round(sum(changes) / len(changes), 4)


def _change_count_frame(
    observation: Observation,
    crop_edges: tuple[int, int, int, int],
) -> Any:
    if _observation_has_change_summary_crop(observation, crop_edges):
        return observation.frame
    return _crop_frame_arc_grid_edges(observation.frame, crop_edges)


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


def _crop_frame_arc_grid_edges(
    frame: Any,
    crop_edges: tuple[int, int, int, int],
) -> Any:
    if crop_edges == (0, 0, 0, 0):
        return frame

    import numpy as np

    array = np.asarray(frame)
    if array.ndim < 2:
        return frame

    left, top, right, bottom = crop_edges
    height, width = int(array.shape[0]), int(array.shape[1])
    box = (
        _scaled_arc_edge(left, width),
        _scaled_arc_edge(top, height),
        width - _scaled_arc_edge(right, width),
        height - _scaled_arc_edge(bottom, height),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return array[:0, :0]
    return array[box[1] : box[3], box[0] : box[2]]


def _scaled_arc_edge(edge: int, axis_size: int) -> int:
    return int(edge * axis_size / ARC_GRID_SIZE + 0.5)


def _frame_surface_size_for_count(frame: Any) -> int:
    import numpy as np

    return _frame_surface_size(np.asarray(frame))


def _action_mode(value: str | None) -> Literal["probing", "policy"] | None:
    if value in {"probing", "policy"}:
        return value
    return None


def unroll_observation(
    observation: Observation,
) -> tuple[Observation, ...]:
    """Normalize one environment observation into ordered frame turns."""

    frames = observation.frames
    if not frames:
        frames = (observation.frame,)
    input_frame_count = len(frames)
    selected_frames = _select_distinct_frames(frames)

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


def bundle_frame_observations(observation: Observation) -> tuple[Observation, ...]:
    """Return deduplicated environment bundle frames as observations."""

    frames = observation.frames or (observation.frame,)
    selected_frames = _select_distinct_frames(frames)
    if len(frames) == 1:
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


def change_summary_transition_frame_observations(
    *,
    previous_observation: Observation,
    next_observation: Observation,
    crop_edges: Any | None,
) -> tuple[Observation, ...]:
    """Return frozen previous-to-current frames for change-summary evidence.

    Previous and final frames are always retained. Only intermediate animation
    frames are dropped when they are consecutive duplicates after applying the
    change-summary ARC-grid crop.
    """

    candidates = [
        _ObservationCandidate(
            observation=_snapshot_observation_frame(
                previous_observation,
                crop_edges=crop_edges,
            ),
            original_index=-1,
        )
    ]
    frames = next_observation.frames or (next_observation.frame,)
    candidates.extend(
        _ObservationCandidate(
            observation=_snapshot_bundle_frame_observation(
                next_observation,
                frame=frame,
                original_index=index,
                input_frame_count=len(frames),
                crop_edges=crop_edges,
            ),
            original_index=index,
        )
        for index, frame in enumerate(frames)
        if frame is not None
    )
    return _dedupe_transition_observations_by_visible_crop(
        tuple(candidates),
    )


def change_summary_observation_snapshot(
    observation: Observation,
    *,
    crop_edges: Any | None,
) -> Observation:
    """Return a stable image snapshot for change-summary transition evidence."""

    return _snapshot_observation_frame(observation, crop_edges=crop_edges)


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


def _select_distinct_frames(frames: tuple[Any, ...]) -> tuple[_SelectedFrame, ...]:
    """Return retained bundle frames after exact consecutive duplicate filtering."""

    candidates = _drop_left_duplicate_frames(frames)
    if not candidates:
        return ()

    selected: list[_SelectedFrame] = []
    last_retained_index = -1
    for candidate in candidates:
        selected_frame = _selected_frame_since(
            candidate,
            last_retained_index=last_retained_index,
        )
        selected.append(selected_frame)
        last_retained_index = candidate.original_index

    return tuple(selected)


def _dedupe_transition_observations_by_visible_crop(
    candidates: tuple[_ObservationCandidate, ...],
) -> tuple[Observation, ...]:
    selected: list[tuple[Observation, int, int]] = []
    last_visible_frame: Any | None = None
    last_retained_index = -1
    final_position = len(candidates) - 1
    for position, candidate in enumerate(candidates):
        visible_frame = observation_to_pil_image(candidate.observation)
        retain_boundary_frame = position == 0 or position == final_position
        if (
            not retain_boundary_frame
            and last_visible_frame is not None
            and _frames_equal(last_visible_frame, visible_frame)
        ):
            continue
        skipped = (
            0
            if candidate.original_index < 0
            else max(0, candidate.original_index - last_retained_index - 1)
        )
        selected.append((candidate.observation, candidate.original_index, skipped))
        last_visible_frame = visible_frame
        last_retained_index = candidate.original_index

    count = len(selected)
    return tuple(
        _with_transition_frame_metadata(
            observation,
            frame_index=index,
            frame_count=count,
            skipped_intermediate_animation_frame_count=skipped,
        )
        for index, (observation, _original_index, skipped) in enumerate(selected)
    )


def _snapshot_observation_frame(
    observation: Observation,
    *,
    crop_edges: Any | None,
) -> Observation:
    image, normalized_crop_edges = _change_summary_snapshot_image(
        observation,
        crop_edges=crop_edges,
    )
    return Observation(
        id=observation.id,
        step=observation.step,
        frame=image,
        frames=(image,),
        raw_frame_data=observation.raw_frame_data,
        metadata={
            **observation.metadata,
            "change_summary_crop_edges": normalized_crop_edges,
        },
    )


def _snapshot_bundle_frame_observation(
    observation: Observation,
    *,
    frame: Any,
    original_index: int,
    input_frame_count: int,
    crop_edges: Any | None,
) -> Observation:
    normalized_crop_edges = normalize_arc_grid_crop_edges(crop_edges)
    image = crop_image_arc_grid_edges(
        frame_to_pil_image(
            frame,
            step=observation.step,
            label=f"{observation.id}-frame-{original_index}",
        ),
        normalized_crop_edges,
    ).copy()
    return Observation(
        id=f"{observation.id}-frame-{original_index}",
        step=observation.step,
        frame=image,
        frames=(image,),
        raw_frame_data=observation.raw_frame_data,
        metadata={
            **observation.metadata,
            "bundle_observation_id": observation.id,
            "bundle_frame_index": original_index,
            "input_frame_count": input_frame_count,
            "change_summary_crop_edges": normalized_crop_edges,
        },
    )


def _change_summary_snapshot_image(
    observation: Observation,
    *,
    crop_edges: Any | None,
) -> tuple[Any, tuple[int, int, int, int]]:
    normalized_crop_edges = normalize_arc_grid_crop_edges(crop_edges)
    image = observation_to_pil_image(observation)
    if _observation_has_change_summary_crop(observation, normalized_crop_edges):
        return image.copy(), normalized_crop_edges
    return (
        crop_image_arc_grid_edges(image, normalized_crop_edges).copy(),
        normalized_crop_edges,
    )


def _observation_has_change_summary_crop(
    observation: Observation,
    crop_edges: tuple[int, int, int, int],
) -> bool:
    metadata_edges = observation.metadata.get("change_summary_crop_edges")
    if metadata_edges is None:
        return False
    try:
        return tuple(metadata_edges) == crop_edges
    except TypeError:
        return False


def _with_transition_frame_metadata(
    observation: Observation,
    *,
    frame_index: int,
    frame_count: int,
    skipped_intermediate_animation_frame_count: int,
) -> Observation:
    return Observation(
        id=observation.id,
        step=observation.step,
        frame=observation.frame,
        frames=observation.frames,
        raw_frame_data=observation.raw_frame_data,
        metadata={
            **observation.metadata,
            "frame_index": frame_index,
            "frame_count": frame_count,
            "skipped_intermediate_animation_frame_count": (
                skipped_intermediate_animation_frame_count
            ),
        },
    )


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
