"""Prompt-facing action history grouping helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionSpec,
)

ANIMATION_NET_NOOP_NOTE = (
    "This action triggered an animation feedback, but previous and current frame "
    "remain identical. So there is no progress but the animation teach you "
    "something."
)


@dataclass(frozen=True, slots=True)
class ActionHistoryGroup:
    """One controllable action plus following animation evidence."""

    action: ActionHistoryEntry | None
    animations: tuple[ActionHistoryEntry, ...] = ()


ActionHistoryRow = ActionHistoryGroup | ActionHistoryResetMarker


def model_facing_action_text(
    action: ActionSpec,
    *,
    crop_edges: object | None = None,
) -> str:
    """Render one action for prompts that use normalized visual ACTION6 data."""

    if action.name == "ACTION6":
        if action.data is None:
            return f"{action.name}(x,y normalized_0_1000,target)"
        if action.target is not None and action.target.strip():
            return f"{action.name} target={json.dumps(action.target.strip())}"
        raise ValueError("ACTION6 action history entries require a target")
    if action.data:
        return f"{action.name} {json.dumps(action.data, sort_keys=True)}"
    return action.name


def group_action_history(
    history: Sequence[ActionHistoryItem],
) -> tuple[ActionHistoryRow, ...]:
    """Group animation rows under actions and keep reset rows explicit."""

    rows: list[ActionHistoryRow] = []
    current_action: ActionHistoryEntry | None = None
    current_animations: list[ActionHistoryEntry] = []
    orphan_animations: list[ActionHistoryEntry] = []

    def flush_current_group() -> None:
        nonlocal current_action, current_animations, orphan_animations
        if orphan_animations:
            rows.append(
                ActionHistoryGroup(action=None, animations=tuple(orphan_animations))
            )
            orphan_animations = []
        if current_action is not None:
            rows.append(
                ActionHistoryGroup(
                    action=current_action,
                    animations=tuple(current_animations),
                )
            )
            current_action = None
            current_animations = []

    for entry in history:
        if isinstance(entry, ActionHistoryResetMarker):
            flush_current_group()
            rows.append(entry)
            continue

        if entry.controllable:
            if orphan_animations:
                rows.append(
                    ActionHistoryGroup(action=None, animations=tuple(orphan_animations))
                )
                orphan_animations = []
            if current_action is not None:
                rows.append(
                    ActionHistoryGroup(
                        action=current_action,
                        animations=tuple(current_animations),
                    )
                )
            current_action = entry
            current_animations = []
            continue

        if current_action is None:
            orphan_animations.append(entry)
        else:
            current_animations.append(entry)

    flush_current_group()
    return tuple(rows)


def grouped_action_history_text(
    history: Sequence[ActionHistoryItem],
    *,
    action_text: Callable[[ActionSpec], str],
    numbered: bool,
) -> str:
    """Render action history with animation evidence merged into action rows."""

    if not history:
        return "none"

    latest_item = history[-1]
    groups = group_action_history(history)
    lines: list[str] = []
    if numbered:
        for index, row in enumerate(groups, start=1):
            if isinstance(row, ActionHistoryResetMarker):
                lines.append(
                    _reset_marker_line(
                        row,
                        prefix=f"{index}. ",
                        latest=row is latest_item,
                    )
                )
                continue
            lines.extend(
                _group_lines(
                    row,
                    prefix=f"{index}. ",
                    indent="   ",
                    latest_item=latest_item,
                    action_text=action_text,
                )
            )
        return "\n".join(lines)

    for row in groups:
        if isinstance(row, ActionHistoryResetMarker):
            lines.append(
                _reset_marker_line(
                    row,
                    prefix="- ",
                    latest=row is latest_item,
                )
            )
            continue
        lines.extend(
            _group_lines(
                row,
                prefix="- ",
                indent="  ",
                latest_item=latest_item,
                action_text=action_text,
            )
        )
    return "\n".join(lines)


def _group_lines(
    group: ActionHistoryGroup,
    *,
    prefix: str,
    indent: str,
    latest_item: ActionHistoryItem,
    action_text: Callable[[ActionSpec], str],
) -> list[str]:
    lines: list[str] = []
    if group.action is None:
        lines.append(f"{prefix}animation_without_prior_action:")
        for animation in group.animations:
            lines.append(
                f"{indent}- "
                + action_history_entry_text(
                    animation,
                    latest=animation is latest_item,
                    action_text=action_text,
                )
            )
    else:
        animation = group.animations[-1] if group.animations else None
        lines.append(
            prefix
            + action_history_entry_text(
                group.action,
                latest=group.action is latest_item or animation is latest_item,
                action_text=action_text,
                include_change_evidence=True,
                merged_animation=animation,
            )
        )
    return lines


def action_history_entry_text(
    entry: ActionHistoryEntry,
    *,
    latest: bool = False,
    action_text: Callable[[ActionSpec], str],
    include_change_evidence: bool = True,
    merged_animation: ActionHistoryEntry | None = None,
) -> str:
    """Render one raw action history entry."""

    text = (
        ""
        if (not entry.controllable and entry.action.is_none())
        else action_text(entry.action)
    )
    if latest and merged_animation is not None:
        text += " [latest]"
    if merged_animation is not None:
        text = f"{text} {_animation_marker_text(merged_animation)}"
    elif not entry.controllable:
        animation_text = _animation_marker_text(entry)
        text = f"{text} {animation_text}" if text else animation_text
    if latest and merged_animation is None:
        text += " [latest]"
    if entry.controllable and entry.action_mode is not None:
        text += f" [mode={entry.action_mode}]"
    if entry.completed_levels is not None:
        text += f" [completed_levels={entry.completed_levels}]"
    if entry.controllable and entry.action_count is not None:
        text += f" [action_count={entry.action_count}]"
    if include_change_evidence:
        evidence_entry = merged_animation or entry
        if evidence_entry.avg_changed_pixel_count is not None:
            text += (
                " [animation_avg_changed_pixels="
                + _changed_pixel_percentage_text(
                    evidence_entry.avg_changed_pixel_count
                )
                + "]"
            )
        else:
            text += (
                f" [changed_pixels={_changed_pixel_percentage_text(evidence_entry)}]"
            )
        summary = evidence_entry.change_summary.strip()
        if (
            evidence_entry.changed_pixel_count == 0
            and evidence_entry.avg_changed_pixel_count is None
        ):
            summary = (
                "No changes happened for this transition. "
                "The previous and current frames are identical"
            )
        if merged_animation is not None and entry.changed_pixel_count == 0:
            text = _append_animation_net_noop_summary(text, summary)
        elif summary:
            text += f" Elements and associated changes:\n{summary}"
    return text


def _animation_marker_text(entry: ActionHistoryEntry) -> str:
    if entry.animation_frame_count is not None:
        return f"[animation: {entry.animation_frame_count} frames]"
    return "[animation]"


def _append_animation_net_noop_summary(text: str, summary: str) -> str:
    note = (
        f"{ANIMATION_NET_NOOP_NOTE} "
        "Elements and associated animation feedback changes:"
    )
    summary = summary.strip()
    if summary:
        return f"{text} {note}\n{summary}"
    return f"{text} {note}"


def _changed_pixel_percentage_text(entry: ActionHistoryEntry | float) -> str:
    value = (
        entry.changed_pixel_count
        if isinstance(entry, ActionHistoryEntry)
        else entry
    )
    if value == 0:
        return "0%"
    return f"{value:.4f}".rstrip("0").rstrip(".") + "%"


def _reset_marker_line(
    marker: ActionHistoryResetMarker,
    *,
    prefix: str,
    latest: bool,
) -> str:
    text = "GAME_RESET"
    if latest:
        text += " [latest]"
    text += f" [reason={marker.reason}] [restart_count={marker.restart_count}]"
    return prefix + text
