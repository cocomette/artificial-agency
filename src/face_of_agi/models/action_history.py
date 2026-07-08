"""Prompt-facing action history grouping helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from face_of_agi.contracts import (
    ActionHistoryEntry,
    ActionHistoryItem,
    ActionHistoryResetMarker,
    ActionHistoryScoreAdvanceMarker,
    ActionSpec,
)
from face_of_agi.models.arc_grid_crop import arc_grid_to_normalized_1000
from face_of_agi.models.change.adapter import format_changed_pixel_percent


@dataclass(frozen=True, slots=True)
class ActionHistoryGroup:
    """One controllable action plus following animation evidence."""

    action: ActionHistoryEntry | None
    animations: tuple[ActionHistoryEntry, ...] = ()


ActionHistoryRow = (
    ActionHistoryGroup
    | ActionHistoryResetMarker
    | ActionHistoryScoreAdvanceMarker
)


def model_facing_action_text(
    action: ActionSpec,
    *,
    crop_edges: Any | None = None,
) -> str:
    """Render one action for prompts that use normalized visual ACTION6 data."""

    if action.name == "ACTION6":
        if action.data is None:
            return f"{action.name}(x,y normalized_0_1000)"
        return (
            f"{action.name} "
            + json.dumps(
                _normalized_action6_data(action.data, crop_edges=crop_edges),
                sort_keys=True,
            )
        )
    if action.data:
        return f"{action.name} {json.dumps(action.data, sort_keys=True)}"
    return action.name


def _normalized_action6_data(
    data: dict[str, Any],
    *,
    crop_edges: Any | None,
) -> dict[str, int]:
    return {
        "x": arc_grid_to_normalized_1000(data, "x", crop_edges=crop_edges),
        "y": arc_grid_to_normalized_1000(data, "y", crop_edges=crop_edges),
    }


def model_facing_action_text_for_crop(
    crop_edges: Any | None,
) -> Callable[[ActionSpec], str]:
    """Return an action renderer for one model-visible crop configuration."""

    return lambda action: model_facing_action_text(
        action,
        crop_edges=crop_edges,
    )


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
        if isinstance(
            entry,
            (ActionHistoryResetMarker, ActionHistoryScoreAdvanceMarker),
        ):
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
    latest_description: str | None = None,
) -> str:
    """Render action history with animation rows nested under action rows."""

    if not history:
        return "none"

    latest_item = history[-1]
    groups = group_action_history(history)
    lines: list[str] = []
    if numbered:
        if latest_description is None:
            raise ValueError("numbered action history requires latest_description")
        lines.append(latest_description)
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
            if isinstance(row, ActionHistoryScoreAdvanceMarker):
                lines.append(
                    _score_advance_marker_line(
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
        if isinstance(row, ActionHistoryScoreAdvanceMarker):
            lines.append(
                _score_advance_marker_line(
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
    else:
        lines.append(
            prefix
            + action_history_entry_text(
                group.action,
                latest=group.action is latest_item,
                action_text=action_text,
            )
        )
    if group.animations:
        lines.append(f"{indent}animation_after:")
        for animation in group.animations:
            lines.append(
                f"{indent}- "
                + action_history_entry_text(
                    animation,
                    latest=animation is latest_item,
                    action_text=action_text,
                )
            )
    return lines


def action_history_entry_text(
    entry: ActionHistoryEntry,
    *,
    latest: bool = False,
    action_text: Callable[[ActionSpec], str],
) -> str:
    """Render one raw action history entry."""

    text = action_text(entry.action)
    if not entry.controllable:
        text += " [animation]"
    retained_count = max(0, entry.retained_animation_frame_count)
    if retained_count > 1:
        text += f" [retained_animation_frames={retained_count}]"
    skipped_count = max(0, entry.skipped_intermediate_animation_frame_count)
    if skipped_count:
        text += f" [skipped_intermediate_animation_frames={skipped_count}]"
    if entry.animation_avg_changed_pixel_percent is not None:
        text += (
            " [animation_avg_changed_pixel_percent="
            f"{format_changed_pixel_percent(entry.animation_avg_changed_pixel_percent)}]"
        )
    if latest:
        text += " [latest]"
    text += (
        " [changed_pixel_percent="
        f"{format_changed_pixel_percent(entry.changed_pixel_percent)}]"
    )
    summary = entry.change_summary.strip()
    if summary:
        text += f" change: {summary}"
    return text


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


def _score_advance_marker_line(
    marker: ActionHistoryScoreAdvanceMarker,
    *,
    prefix: str,
    latest: bool,
) -> str:
    text = "SCORE_ADVANCE"
    if latest:
        text += " [latest]"
    text += (
        f" [previous_score={_nullable_metric_text(marker.previous_score)}]"
        f" [new_score={_nullable_metric_text(marker.new_score)}]"
        f" [delta={_nullable_metric_text(marker.delta)}]"
    )
    return prefix + text


def _nullable_metric_text(value: float | None) -> str:
    if value is None:
        return "null"
    return str(value)
