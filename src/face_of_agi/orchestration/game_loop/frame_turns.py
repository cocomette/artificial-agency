"""Frame-turn helpers for the online learner loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionSpec,
    FrameControlMode,
    Observation,
)

_NO_ANCHOR = object()


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
        raise RuntimeError(
            f"online learner returned invalid action for current frame: {action.name}"
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


def _select_animation_keyframes(
    frames: tuple[Any, ...],
    *,
    threshold: int,
    anchor_frame: Any,
) -> tuple[_SelectedFrame, ...]:
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
    import numpy as np

    return np.issubdtype(array.dtype, np.number)


def _rgb_like_array(array: Any) -> bool:
    return array.ndim == 3 and array.shape[-1] in {3, 4}


def _frame_surface_size(array: Any) -> int:
    if array.shape == ():
        return 1
    if _rgb_like_array(array):
        return int(array.shape[0] * array.shape[1])
    return int(array.size)


def _structurally_equal(left: Any, right: Any) -> bool:
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
